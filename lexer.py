#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lexer.py —— 纯 Python 标准库的单文件词法分析工具。

用法：
    python3 lexer.py 规则文件 待分析文本文件
    python3 lexer.py --demo          # 运行内置自测样例

规则文件格式（每行一条，# 开头为注释，空行忽略）：

    类型  /正则/                     单记号：正则匹配（必须从当前位置开始匹配）
    类型  "字面量"                   单记号：字面量匹配
    类型  "起始标记" ... "结束标记"   配对记号：从起始标记到结束标记（可跨行），
                                     起止之间全部内容归该记号

说明：
    - 引号内支持转义：\\n \\t \\r \\0 \\\\ \\" \\'
    - 多个规则在同一位置匹配时，选最长匹配；长度相同按规则定义顺序取舍
    - 字符串、注释等配对记号内部的内容不会被其他规则重复匹配
    - 类型名为 SKIP 的规则只消耗输入、不输出记号（常用于空白）
    - 起始标记出现但结束标记缺失时，报告起始所在行号
    - 规则语法错误（缺标记、模式残缺、正则无效等）会逐行报告
    - 没有规则匹配的字符会逐字符报告并跳过

退出码：0 = 无错误；1 = 分析过程有错误；2 = 规则文件错误。
"""

import argparse
import bisect
import re
import sys
from collections import namedtuple

Token = namedtuple("Token", ["type", "line", "content"])

SKIP_TYPE = "SKIP"


class RuleError(Exception):
    """单条规则的语法错误。"""


class Rule:
    __slots__ = ("type", "kind", "start", "end", "regex", "lineno")

    def __init__(self, type_, kind, start=None, end=None, regex=None, lineno=0):
        self.type = type_
        self.kind = kind          # "literal" | "regex" | "pair"
        self.start = start        # literal / pair 的起始标记
        self.end = end            # pair 的结束标记
        self.regex = regex        # regex 的编译结果
        self.lineno = lineno


_ESCAPES = {
    "n": "\n", "t": "\t", "r": "\r", "0": "\0",
    "\\": "\\", '"': '"', "'": "'", "/": "/",
}


def _parse_quoted(s, i):
    """从 s[i]（必须是双引号）解析一个字面量，返回 (值, 结束后的下标)。"""
    out = []
    i += 1
    while i < len(s):
        c = s[i]
        if c == "\\":
            if i + 1 >= len(s):
                raise RuleError("转义字符不完整")
            e = s[i + 1]
            if e not in _ESCAPES:
                raise RuleError("无法识别的转义序列 \\%s" % e)
            out.append(_ESCAPES[e])
            i += 2
        elif c == '"':
            return "".join(out), i + 1
        else:
            out.append(c)
            i += 1
    raise RuleError("引号未闭合，字面量残缺")


def _parse_regex_body(s):
    """s[0] 必须是 '/'，返回 (正则文本, 结束后的下标)。"""
    i = 1
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "/":
            return s[1:i], i + 1
        i += 1
    raise RuleError("正则模式缺少结束的 /")


def parse_rules(rules_text):
    """解析规则文本，返回 (规则列表, 错误列表)。有错的规定跳过，其余仍可用。"""
    rules, errors = [], []
    for lineno, raw in enumerate(rules_text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"\S+", line)
        type_ = m.group(0)
        rest = line[m.end():].strip()
        if not rest:
            errors.append("规则第 %d 行：记号类型 %r 缺少匹配模式" % (lineno, type_))
            continue
        try:
            if rest[0] == '"':
                start, j = _parse_quoted(rest, 0)
                tail = rest[j:].strip()
                if tail.startswith("..."):
                    tail2 = tail[3:].strip()
                    if not tail2 or tail2[0] != '"':
                        raise RuleError("配对模式缺少结束标记（应为 \"起始\" ... \"结束\"）")
                    end, k = _parse_quoted(tail2, 0)
                    if tail2[k:].strip():
                        raise RuleError("结束标记后有多余内容")
                    if not start or not end:
                        raise RuleError("起始/结束标记不能为空")
                    rules.append(Rule(type_, "pair", start=start, end=end, lineno=lineno))
                elif tail:
                    raise RuleError("字面量后有多余内容：%r" % tail)
                else:
                    if not start:
                        raise RuleError("匹配模式不能为空")
                    rules.append(Rule(type_, "literal", start=start, lineno=lineno))
            elif rest[0] == "/":
                pattern, j = _parse_regex_body(rest)
                if rest[j:].strip():
                    raise RuleError("正则模式后有多余内容")
                try:
                    compiled = re.compile(pattern)
                except re.error as exc:
                    raise RuleError("正则表达式无效：%s" % exc)
                rules.append(Rule(type_, "regex", regex=compiled, lineno=lineno))
            else:
                raise RuleError("无法识别的模式 %r（应以 \" 或 / 开头）" % rest)
        except RuleError as exc:
            errors.append("规则第 %d 行：%s" % (lineno, exc))
    return rules, errors


def _match_rule(rule, text, pos):
    """在 pos 处尝试匹配规则，返回 (匹配长度, 错误消息或 None)。长度 0 表示不匹配。"""
    if rule.kind == "literal":
        if text.startswith(rule.start, pos):
            return len(rule.start), None
        return 0, None
    if rule.kind == "regex":
        m = rule.regex.match(text, pos)
        if m and m.end() > pos:
            return m.end() - pos, None
        return 0, None
    # pair：起始标记命中后，向后找结束标记，中间全部归该记号（可跨行）
    if not text.startswith(rule.start, pos):
        return 0, None
    close = text.find(rule.end, pos + len(rule.start))
    if close == -1:
        return len(text) - pos, "记号 %s 的起始标记 %r 出现后缺少结束标记 %r" % (
            rule.type, rule.start, rule.end)
    return close + len(rule.end) - pos, None


def lex(text, rules):
    """扫描文本，返回 (记号列表, 错误列表)。"""
    tokens, errors = [], []
    line_starts = [0] + [m.end() for m in re.finditer("\n", text)]

    def line_of(p):
        return bisect.bisect_right(line_starts, p)

    pos, n = 0, len(text)
    while pos < n:
        best_len, best_rule, best_err = 0, None, None
        for rule in rules:  # 严格更长才替换：长度相同保留先定义的规则
            length, err = _match_rule(rule, text, pos)
            if length > best_len:
                best_len, best_rule, best_err = length, rule, err
        if best_rule is None:
            errors.append("第 %d 行：没有规则能匹配字符 %r" % (line_of(pos), text[pos]))
            pos += 1
            continue
        if best_err:
            errors.append("第 %d 行：%s" % (line_of(pos), best_err))
        if best_rule.type != SKIP_TYPE:
            tokens.append(Token(best_rule.type, line_of(pos), text[pos:pos + best_len]))
        pos += best_len
    return tokens, errors


def _shorten(s, limit=50):
    r = repr(s)
    return r if len(r) <= limit else r[:limit - 3] + "..."


def print_report(tokens, errors, out=None):
    out = out or sys.stdout
    print("记号流（共 %d 个）：" % len(tokens), file=out)
    print("  %-6s%-14s%s" % ("行号", "类型", "内容"), file=out)
    for t in tokens:
        print("  %-6d%-14s%s" % (t.line, t.type, _shorten(t.content)), file=out)
    print("错误报告：", file=out)
    if errors:
        for e in errors:
            print("  " + e, file=out)
    else:
        print("  （无错误）", file=out)


# ---------------------------------------------------------------- 自测样例

DEMO_RULES = r'''
# 空白：匹配但不输出
SKIP        /[ \t\r\n]+/
# 配对记号：块注释 / 行注释 / 字符串，起止之间可跨行，内容不被其他规则匹配
COMMENT     "/*" ... "*/"
LINECOMMENT "//" ... "\n"
STRING      "\"" ... "\""
# 关键字先于标识符定义：长度相同时按定义顺序，KEYWORD 取胜
KEYWORD     /if|else|while|return/
NUMBER      /\d+(\.\d+)?/
IDENT       /[A-Za-z_][A-Za-z0-9_]*/
OP          /==|!=|<=|>=|[-+*\/<>=;(){}]/
'''

DEMO_TEXT = """\
if x1 >= 42 then
/* 多行
   注释 */
say("hi") // 行注释
"""

ERR_TEXT_UNTERMINATED = 'name = "这个字符串没有结束'
ERR_TEXT_UNMATCHED = "a @ b"

BAD_RULES = r'''
NO_PATTERN
PAIR_NO_END "/*" ...
BAD_REGEX /[0-9/
UNKNOWN ???
EMPTY ""
'''


def run_demo():
    print("=" * 60)
    print("样例 1：正常分析（含跨行注释、字符串、最长匹配、同长按序）")
    print("=" * 60)
    rules, rule_errors = parse_rules(DEMO_RULES)
    assert not rule_errors, rule_errors
    tokens, errors = lex(DEMO_TEXT, rules)
    print_report(tokens, errors)

    print()
    print("=" * 60)
    print("样例 2：起始标记出现但结束标记缺失")
    print("=" * 60)
    tokens2, errors2 = lex(ERR_TEXT_UNTERMINATED, rules)
    print_report(tokens2, errors2)

    print()
    print("=" * 60)
    print("样例 3：存在没有规则匹配的字符")
    print("=" * 60)
    tokens3, errors3 = lex(ERR_TEXT_UNMATCHED, rules)
    print_report(tokens3, errors3)

    print()
    print("=" * 60)
    print("样例 4：规则语法错误")
    print("=" * 60)
    _, bad_errors = parse_rules(BAD_RULES)
    for e in bad_errors:
        print("  " + e)

    _self_check(rules, tokens, errors, errors2, errors3, bad_errors)
    print()
    print("自测断言全部通过。")


def _self_check(rules, tokens, errors, errors2, errors3, bad_errors):
    types = [t.type for t in tokens]
    assert types == ["KEYWORD", "IDENT", "OP", "NUMBER", "IDENT",
                     "COMMENT", "IDENT", "OP", "STRING", "OP",
                     "LINECOMMENT"], types
    assert tokens[0].content == "if" and tokens[0].type == "KEYWORD"  # 同长按定义顺序
    assert tokens[2].content == ">="                                    # 最长匹配
    assert tokens[5].line == 2 and "\n" in tokens[5].content            # 跨行配对
    assert tokens[8].line == 4 and tokens[8].content == '"hi"'
    assert errors == []
    assert any("第 1 行" in e and "结束标记" in e for e in errors2), errors2
    assert any("'@'" in e for e in errors3), errors3
    assert len(bad_errors) == 5, bad_errors


def main(argv):
    parser = argparse.ArgumentParser(
        description="纯标准库词法分析工具：输出记号流与错误报告")
    parser.add_argument("rules", nargs="?", help="规则文件路径")
    parser.add_argument("text", nargs="?", help="待分析文本文件路径")
    parser.add_argument("--demo", action="store_true", help="运行内置自测样例")
    args = parser.parse_args(argv)

    if args.demo:
        run_demo()
        return 0
    if not args.rules or not args.text:
        parser.error("需要提供规则文件和待分析文本文件（或用 --demo 运行自测）")

    with open(args.rules, encoding="utf-8") as f:
        rules_text = f.read()
    with open(args.text, encoding="utf-8") as f:
        text = f.read()

    rules, rule_errors = parse_rules(rules_text)
    for e in rule_errors:
        print(e, file=sys.stderr)
    if rule_errors:
        print("规则文件存在错误，已终止。", file=sys.stderr)
        return 2

    tokens, errors = lex(text, rules)
    print_report(tokens, errors)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单文件词法分析器（仅依赖 Python 标准库）。

用法：
    python3 lexer.py 规则文件 待分析文本文件
    python3 lexer.py --selftest          # 运行内嵌自测样例

规则文件格式（每行一条规则，空行与 # 开头的行被忽略）：
    记号类型 LIT 字面量                  —— 单记号精确匹配
    记号类型 RE  正则表达式              —— 单记号正则匹配（锚定在当前位置）
    记号类型 BEGIN 起始标记 END 结束标记 —— 定界记号，可跨行，
                                           起始到结束之间全部归该记号

特殊记号类型 SKIP：匹配成功但不输出记号（用于空白、空白注释等）。
LIT/BEGIN/END 标记中的转义：\\n 换行、\\t 制表符、\\s 空格、\\\\ 反斜杠；
RE 模式不做转义处理（正则自身的 \\d、\\s 等原样生效，空格请用 \\s 或 [ ]）。

冲突解决：同一位置多个规则匹配时取最长匹配；长度相同取先定义的规则。
"""

import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Rule:
    kind: str                        # 记号类型名
    mode: str                        # 'lit' | 're' | 'span'
    begin: str                       # lit: 字面量；re: 正则原文；span: 起始标记
    end: Optional[str] = None        # span: 结束标记
    regex: Optional["re.Pattern"] = None

    def match(self, text: str, pos: int) -> Optional[Tuple[int, Optional[str]]]:
        """在 pos 处尝试匹配。返回 (匹配长度, 错误标记或 None)；不匹配返回 None。"""
        if self.mode == "lit":
            if text.startswith(self.begin, pos):
                return (len(self.begin), None)
            return None
        if self.mode == "re":
            m = self.regex.match(text, pos)
            if m:
                return (m.end() - pos, None)
            return None
        # span：起始标记命中后，结束标记之间（含）全部归该记号，可跨行
        if text.startswith(self.begin, pos):
            j = text.find(self.end, pos + len(self.begin))
            if j < 0:
                return (len(text) - pos, "unterminated")
            return (j + len(self.end) - pos, None)
        return None


@dataclass
class Token:
    kind: str
    line: int                        # 起始行号（从 1 开始）
    text: str


def _unescape(s: str) -> str:
    out, i = [], 0
    mapping = {"n": "\n", "t": "\t", "r": "\r", "s": " ", "\\": "\\"}
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            out.append(mapping.get(s[i + 1], s[i + 1]))
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def parse_rules(rules_text: str) -> Tuple[List[Rule], List[str]]:
    """解析规则文本，返回 (规则列表, 规则语法错误列表)。"""
    rules: List[Rule] = []
    errors: List[str] = []
    for lineno, raw in enumerate(rules_text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            errors.append(f"规则第{lineno}行：缺少匹配模式：{line!r}")
            continue
        kind, directive = parts[0], parts[1].upper()
        if directive == "LIT":
            if len(parts) != 3:
                errors.append(f"规则第{lineno}行：LIT 模式残缺，应为：{kind} LIT 字面量")
                continue
            lit = _unescape(parts[2])
            if not lit:
                errors.append(f"规则第{lineno}行：LIT 字面量不能为空")
                continue
            rules.append(Rule(kind, "lit", lit))
        elif directive == "RE":
            if len(parts) != 3:
                errors.append(f"规则第{lineno}行：RE 模式残缺，应为：{kind} RE 正则表达式")
                continue
            pattern = parts[2]  # 正则不转义，保留 \d、\s 等
            try:
                regex = re.compile(pattern)
            except re.error as exc:
                errors.append(f"规则第{lineno}行：正则表达式错误：{exc}")
                continue
            if regex.match("") is not None:
                errors.append(f"规则第{lineno}行：正则可匹配空串（会导致死循环）：{pattern!r}")
                continue
            rules.append(Rule(kind, "re", pattern, regex=regex))
        elif directive == "BEGIN":
            if len(parts) != 5 or parts[3].upper() != "END":
                errors.append(
                    f"规则第{lineno}行：定界模式残缺，应为：{kind} BEGIN 起始标记 END 结束标记"
                )
                continue
            begin, end = _unescape(parts[2]), _unescape(parts[4])
            if not begin or not end:
                errors.append(f"规则第{lineno}行：起始/结束标记不能为空")
                continue
            rules.append(Rule(kind, "span", begin, end=end))
        else:
            errors.append(
                f"规则第{lineno}行：未知模式类型 {parts[1]!r}（应为 LIT / RE / BEGIN ... END ...）"
            )
    return rules, errors


def tokenize(text: str, rules: List[Rule]) -> Tuple[List[Token], List[str]]:
    """对 text 做词法分析，返回 (记号流, 词法错误列表)。"""
    tokens: List[Token] = []
    errors: List[str] = []
    pos, line, n = 0, 1, len(text)
    while pos < n:
        best_len, best_rule, best_err = -1, None, None
        for rule in rules:  # 按定义顺序扫描；严格大于才替换 => 等长取先定义者
            res = rule.match(text, pos)
            if res is None:
                continue
            length, err = res
            if length <= 0:
                continue
            if length > best_len:
                best_len, best_rule, best_err = length, rule, err
        if best_rule is None:
            ch = text[pos]
            errors.append(f"第{line}行：没有规则匹配的字符 {ch!r}")
            if ch == "\n":
                line += 1
            pos += 1
            continue
        content = text[pos:pos + best_len]
        if best_err == "unterminated":
            errors.append(
                f"第{line}行：{best_rule.kind} 的起始标记 {best_rule.begin!r} "
                f"缺少结束标记 {best_rule.end!r}"
            )
        if best_rule.kind != "SKIP":
            tokens.append(Token(best_rule.kind, line, content))
        line += content.count("\n")
        pos += best_len
    return tokens, errors


def format_report(tokens: List[Token], errors: List[str]) -> str:
    lines = ["记号流："]
    if tokens:
        for t in tokens:
            lines.append(f"  行{t.line:<4} {t.kind:<12} {t.text!r}")
    else:
        lines.append("  （无）")
    lines.append("错误报告：")
    if errors:
        lines.extend(f"  {e}" for e in errors)
    else:
        lines.append("  （无）")
    return "\n".join(lines)


# ---------------------------------------------------------------- 自测样例

DEMO_RULES = r"""
# 注释与空行会被忽略
SKIP     RE \s+
STRING   BEGIN " END "
COMMENT  BEGIN /* END */
ASSIGN   LIT ==
EQ       LIT =
IF       LIT if
ID       RE [A-Za-z_][A-Za-z0-9_]*
NUMBER   RE [0-9]+
"""

DEMO_INPUT = """if x == 42 /* 多行
注释 = == "不算字符串" */ ifx
"未闭合字符串
@"""

BAD_RULES = r"""
A LIT
B BEGIN "
C RE [
D RE a*
E BLAH x
OK LIT ok
"""


def run_selftest() -> int:
    failures = []

    # 1) 规则语法错误报告
    rules_bad, rule_errors = parse_rules(BAD_RULES)
    assert len(rules_bad) == 1 and rules_bad[0].kind == "OK", rules_bad
    assert len(rule_errors) == 5, rule_errors
    print("== 用例1：规则语法错误 ==")
    for e in rule_errors:
        print("  " + e)

    # 2) 主样例：最长匹配、同长取先定义、跨行定界、未闭合、无匹配字符
    rules, rerrs = parse_rules(DEMO_RULES)
    assert not rerrs, rerrs
    tokens, terrs = tokenize(DEMO_INPUT, rules)
    print("\n== 用例2：完整分析 ==")
    print(format_report(tokens, rerrs + terrs))

    kinds = [t.kind for t in tokens]
    # ASSIGN(==) 胜过 EQ(=)：最长匹配
    assert kinds[:4] == ["IF", "ID", "ASSIGN", "NUMBER"], kinds
    # ifx：ID(3) 比 IF(2) 长 => ID；而独立 if 与 ID 等长 => 先定义的 IF
    assert tokens[0].text == "if" and tokens[0].kind == "IF"
    assert any(t.kind == "ID" and t.text == "ifx" for t in tokens)
    # 注释跨行，且其内部的 = == "..." 不产生其他记号
    comment = next(t for t in tokens if t.kind == "COMMENT")
    assert comment.line == 1 and "\n" in comment.text and '="不算字符串"' not in comment.text
    assert not any(t.text == "=" for t in tokens), tokens
    # 字符串未闭合：报告起始行号（第3行）
    assert any("第3行" in e and "缺少结束标记" in e for e in terrs), terrs
    # 字符串内容（第4行的 @ 在字符串内部）不产生“无匹配字符”错误
    assert not any("@" in e for e in terrs), terrs

    # 3) 无规则匹配字符报错
    _, terrs2 = tokenize("a@b", [r for r in rules if r.kind == "ID"])
    assert any("第1行" in e and "'@'" in e for e in terrs2), terrs2
    print("\n== 用例3：无匹配字符 == ")
    for e in terrs2:
        print("  " + e)

    # 4) 行号统计
    toks3, _ = tokenize("if\nif\n==", rules)
    assert [t.line for t in toks3] == [1, 2, 3], toks3
    print("\n== 用例4：行号 ==")
    for t in toks3:
        print(f"  行{t.line} {t.kind} {t.text!r}")

    if failures:
        print("\n自测失败:", failures)
        return 1
    print("\n全部自测通过 ✔")
    return 0


def main(argv: List[str]) -> int:
    if len(argv) == 2 and argv[1] == "--selftest":
        return run_selftest()
    if len(argv) != 3:
        print(__doc__.strip())
        return 2
    with open(argv[1], "r", encoding="utf-8") as f:
        rules_text = f.read()
    with open(argv[2], "r", encoding="utf-8") as f:
        text = f.read()
    rules, rule_errors = parse_rules(rules_text)
    tokens, lex_errors = tokenize(text, rules)
    print(format_report(tokens, rule_errors + lex_errors))
    return 1 if (rule_errors or lex_errors) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

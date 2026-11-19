# -*- coding: utf-8 -*-
"""前端安全渲染纯逻辑测试。

不启动 Streamlit 运行时：用 AST 把 frontend/app.py 里的 safe_md 单独抽出来执行，
确保"代码块外的裸尖括号转义、代码块内原样"这条契约不被破坏。
该契约直接对应线上报红：react-dom 的 removeChild 节点错乱。
"""
from __future__ import annotations

import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "frontend" / "app.py"


def _load_safe_md():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "safe_md")
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<safe_md>", "exec"), ns)
    return ns["safe_md"]


class TestSafeMd:
    def setup_method(self):
        self.f = _load_safe_md()

    def test_bare_angle_brackets_escaped(self):
        assert self.f("<bits/stdc++.h>") == "&lt;bits/stdc++.h&gt;"

    def test_fenced_code_preserved(self):
        text = "```cpp\n#include <bits/stdc++.h>\n```"
        assert self.f(text) == text

    def test_inline_code_preserved_but_body_escaped(self):
        out = self.f("用 `<n>` 表示，正文有 <c1⊕c2>")
        assert "`<n>`" in out
        assert "&lt;c1⊕c2&gt;" in out

    def test_plain_markdown_unchanged(self):
        text = "**结论先行**：复杂度 O(n log n)"
        assert self.f(text) == text

    def test_empty(self):
        assert self.f("") == ""

    def test_stream_uses_plain_text(self):
        """流式渲染必须用 text() 而非 markdown()，从源头避免半成品解析。"""
        src = APP.read_text(encoding="utf-8")
        assert "placeholder.text(text + \" ▌\")" in src
        assert "placeholder.markdown(text + \" ▌\")" not in src

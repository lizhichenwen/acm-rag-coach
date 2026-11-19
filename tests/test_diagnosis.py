# -*- coding: utf-8 -*-
"""代码诊断模块单测：静态特征提取 + 规则候选 + 校验降级。

背景：diagnosis 是任务书「重点需要研究的问题 3」，但此前零测试覆盖。
这里把可解释特征、规则判定、AC 短路、白名单校验都固定下来，
防止重构（如 loop_nesting 的花括号栈写法）回归。

原则：不依赖网络、不依赖数据库、不依赖大模型——毫秒级可跑。
"""
from __future__ import annotations

import pytest

from acm_coach.diagnosis.engine import diagnose, rule_candidates
from acm_coach.diagnosis.features import (
    extract_features, loop_nesting, normalize_language, recursion_used,
)


class TestNormalizeLanguage:
    def test_aliases(self):
        assert normalize_language("C++") == "cpp"
        assert normalize_language("cxx") == "cpp"
        assert normalize_language("PY") == "python"
        assert normalize_language("py") == "python"

    def test_unknown_falls_back_to_cpp(self):
        assert normalize_language("golang") == "cpp"
        assert normalize_language(None) == "cpp"


class TestLoopNesting:
    """回归：if 块的 } 曾错误抵消循环层数，导致 ET07 漏诊。"""

    @pytest.mark.parametrize("code,expected", [
        # 三层纯循环
        ("for (int i=0;i<n;i++){\n for(int j=0;j<n;j++){\n  for(int k=0;k<n;k++){x++;}\n }\n}", 3),
        # 循环体内先出现 if 块，再嵌套循环：旧版算 1，正确是 2
        ("for (int i=0;i<n;i++){\n if(a[i]){x++;}\n for(int j=0;j<n;j++){y++;}\n}", 2),
        # if 块包住两层循环
        ("for (int i=0;i<n;i++){\n if(a[i]){\n  for(int j=0;j<n;j++){y++;}\n }\n}", 2),
        # 单层循环内多个 if，不应累加
        ("for (int i=0;i<n;i++){\n if(a[i]){x++;}\n if(b[i]){y++;}\n}", 1),
        # 同级两个独立循环，不累加
        ("for (int i=0;i<n;i++){x++;}\nfor (int j=0;j<m;j++){y++;}", 1),
        # 无花括号的单语句循环体不产生块嵌套
        ("for (int i=0;i<n;i++) x++;", 0),
        # while 套 for
        ("while(n--){\n for(int i=0;i<n;i++){x++;}\n}", 2),
    ])
    def test_cpp_nesting(self, code, expected):
        assert loop_nesting(code, "cpp") == expected

    def test_python_nesting_by_indent(self):
        code = "for i in range(n):\n    for j in range(n):\n        if a[j]:\n            pass"
        assert loop_nesting(code, "python") == 2

    def test_python_sibling_loops_not_stacked(self):
        code = "for i in range(n):\n    pass\nfor j in range(n):\n    pass"
        assert loop_nesting(code, "python") == 1


class TestExtractFeatures:
    def test_unified_keys_across_languages(self):
        """三种语言必须返回同一套特征键——规则候选直接按下标取值。"""
        code = "for i in range(n):\n    x += a[i]"
        keys = set(extract_features(code, "WA", "python"))
        for lang in ("cpp", "java"):
            assert set(extract_features(code, "WA", lang)) == keys

    def test_tle_nesting_survives_if_blocks(self):
        """核心回归：if 块不应把 3 层循环抵消成 1 层。"""
        code = ("for(int i=0;i<n;i++){\n if(a[i]){x++;}\n"
                " for(int j=0;j<n;j++){\n  if(b[j]){y++;}\n"
                "  for(int k=0;k<n;k++){z++;}\n }\n}")
        assert extract_features(code, "TLE", "cpp")["loop_nesting"] == 3

    def test_cpp_io_features(self):
        f = extract_features("cin>>n; cout<<endl;", "WA", "cpp")
        assert f["has_cin_cout"] is True
        assert f["uses_endl"] is True
        assert f["sync_with_stdio"] is False

    def test_python_has_no_cpp_only_features(self):
        f = extract_features("print(input())", "WA", "python")
        assert f["has_cin_cout"] is False
        assert f["sync_with_stdio"] is False
        assert f["long_long"] is False  # Python 整数无限精度


class TestRuleCandidates:
    def test_deep_loop_gives_et07(self):
        code = ("for(int i=0;i<n;i++){\n if(a[i]){x++;}\n"
                " for(int j=0;j<n;j++){\n  if(b[j]){y++;}\n"
                "  for(int k=0;k<n;k++){z++;}\n }\n}")
        f = extract_features(code, "TLE", "cpp")
        causes = {c["cause_code"] for c in rule_candidates(f)}
        assert "ET07" in causes

    def test_shallow_loop_does_not_give_et07(self):
        f = extract_features("for(int i=0;i<n;i++){ if(a[i]){x++;} }", "TLE", "cpp")
        assert "ET07" not in {c["cause_code"] for c in rule_candidates(f)}


class TestDiagnoseShortCircuit:
    def test_ac_never_invents_error(self):
        """AC 表示全部测试点通过，绝不编造错因（任务书防幻觉要求）。"""
        result = diagnose("int main(){return 0;}", "AC", "A+B", llm=None)
        assert result["cause_code"] == "AC"
        assert result["points"] == []
        assert result["confidence"] == 1.0
        assert result["engine"] == "rule"

    def test_no_llm_falls_back_to_rule(self):
        code = "for(int i=0;i<n;i++){\n for(int j=0;j<n;j++){\n  for(int k=0;k<n;k++){x++;}\n }\n}"
        result = diagnose(code, "TLE", "三层循环", llm=None)
        assert result["engine"] == "rule"
        assert result["cause_code"] == "ET07"

    def test_bad_verdict_defaults_to_wa(self):
        result = diagnose("int a=1; int main(){return a;}", "UNKNOWN", "", llm=None)
        assert result["cause_code"] in {"ET11", "ET14"}  # 规则兜底，不崩溃


class TestRecursionDetection:
    def test_python_self_call(self):
        found, name = recursion_used("def dfs(u):\n    dfs(u+1)", "python")
        assert found and name == "dfs"

    def test_no_recursion(self):
        found, _ = recursion_used("int main(){ return 0; }", "cpp")
        assert found is False

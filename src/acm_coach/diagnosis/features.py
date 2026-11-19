# -*- coding: utf-8 -*-
"""代码静态特征提取（不执行代码），支持 C++ / Python / Java。

为什么先做静态特征：
1. 判题结果是学生手填的（WA/TLE/RE），没有真实运行数据；
2. 静态特征能覆盖"可规则判定"的错因（越界、溢出、递归深度、循环层数），
   让大模型只在"需要语义理解"的错因上出力，降低幻觉与成本；
3. 特征本身可解释，答辩时可以逐条展示"为什么判成复杂度不达标"。

多语言说明：
- 三种语言共用同一套特征键（诊断引擎的规则候选依赖这些键），但提取正则各按语法；
- 语言专属概念（如 C++ 的 sync_with_stdio、Python 的无限精度整数）在不适用的
  语言上保持 False，避免"概念不存在"被误判成"存在风险"。
"""
from __future__ import annotations

import re

_CJK = re.compile(r"[\u4e00-\u9fff]")
SUPPORTED_LANGUAGES = ("cpp", "python", "java")


def normalize_language(language: str | None) -> str:
    """把前端/接口传入的语言归一到受支持集合，未知值一律退化为 cpp。"""
    value = (language or "cpp").strip().lower()
    aliases = {"c++": "cpp", "cxx": "cpp", "py": "python", "golang": "cpp"}
    value = aliases.get(value, value)
    return value if value in SUPPORTED_LANGUAGES else "cpp"


def _strip_comments(code: str, language: str = "cpp") -> str:
    if language == "python":
        code = re.sub(r"\"\"\"[\s\S]*?\"\"\"", "", code)
        code = re.sub(r"'''[\s\S]*?'''", "", code)
        return re.sub(r"#[^\n]*", "", code)
    code = re.sub(r"//[^\n]*", "", code)
    return re.sub(r"/\*.*?\*/", "", code, flags=re.S)


def _brace_depth_profile(code: str, language: str = "cpp") -> tuple[int, list[int]]:
    if language == "python":
        # Python 没有花括号，用最大缩进层级近似代码块嵌套深度（4 空格=1 层）
        levels = []
        for line in code.splitlines():
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            levels.append(indent // 4)
        return (max(levels) if levels else 0), []
    depth = 0
    max_depth = 0
    for ch in code:
        if ch == "{":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == "}":
            depth = max(0, depth - 1)
    return max_depth, []


def loop_nesting(code: str, language: str = "cpp") -> int:
    """近似计算循环嵌套层数。

    C++/Java：按花括号配平，统计同一条链路上的 for/while；
    Python：按缩进层级维护循环栈，统计同一链路上 for/while。
    """
    if language == "python":
        best = 0
        stack: list[int] = []
        for line in code.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))
            while stack and indent <= stack[-1]:
                stack.pop()
            if re.match(r"(for|while)\s+.*:\s*$", stripped):
                stack.append(indent)
                best = max(best, len(stack))
        return best

    # 用花括号栈精确跟踪「同一条链路上的循环块数」：
    # 每个 { 入栈一个标记，表示这个块是不是由 for/while 引入；每个 } 出栈，
    # 只有循环块出栈时才把 current 减 1。
    #
    # 为什么要这么写：旧版只在遇到循环时入栈，却对每一个 } 都出栈并减 current。
    # 于是循环体里任意一个 if 块的 } 都会把循环的层数抵消掉——
    # 「for 里先写 if 再嵌一层 for」本应是 2 层，旧版只算出 1 层，
    # 导致 rule_candidates 的 loop_nesting >= 3 判定失效，ET07（复杂度过高/TLE）
    # 被漏诊，直接影响诊断准确率这个核心指标。
    best = 0
    current = 0
    stack: list[bool] = []
    pending_loop = False  # 已见到 for/while 头，等待它后面的 { 落地成循环块
    paren = 0             # 括号深度，用来区分 for 头里的分号与语句分号
    for line in code.splitlines():
        stripped = line.strip()
        if re.match(r"(for|while)\b", stripped):
            pending_loop = True
        for ch in stripped:
            if ch == "(":
                paren += 1
            elif ch == ")":
                paren = max(0, paren - 1)
            elif ch == "{":
                stack.append(pending_loop)
                if pending_loop:
                    current += 1
                    best = max(best, current)
                pending_loop = False
            elif ch == "}":
                if stack and stack.pop():
                    current = max(0, current - 1)
            elif ch == ";" and paren == 0:
                # for (...) stmt; 这种无花括号的单语句循环体不产生嵌套块
                pending_loop = False
    return best


def recursion_used(code: str, language: str = "cpp") -> tuple[bool, str]:
    """检测是否存在自调用函数（函数体内出现同名标识符 + 括号）。"""
    if language == "python":
        pattern = r"^\s*def\s+(\w+)\s*\([^)]*\)\s*:"
        for match in re.finditer(pattern, code, re.M):
            name = match.group(1)
            body = code[match.end():]
            if re.search(rf"\b{name}\s*\(", body[:4000]):
                return True, name
        return False, ""

    if language == "java":
        # 修饰符 + 返回类型 + 方法名(...){ ，覆盖 public/private/static 等
        pattern = r"(?:public|private|protected|static|\s|\w<[^>]+>)*\b[\w<>\[\]]+\s+(\w+)\s*\([^;]*?\)\s*\{"
    else:
        pattern = r"\b(?:int|long long|void|double|bool|ll)\s+(\w+)\s*\([^;]*?\)\s*\{"

    for match in re.finditer(pattern, code):
        name = match.group(1)
        body = code[match.end():]
        if re.search(rf"\b{name}\s*\(", body[:4000]):
            return True, name
    return False, ""


def extract_features(code: str, verdict: str = "WA", language: str = "cpp") -> dict:
    raw = code or ""
    lang = normalize_language(language)
    clean = _strip_comments(raw, lang)
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    max_depth, _ = _brace_depth_profile(clean, lang)
    nested = loop_nesting(clean, lang)
    recursive, func_name = recursion_used(clean, lang)

    if lang == "python":
        array_sizes = [int(m) for m in re.findall(r"\[\s*0\s*\]\s*\*\s*(\d{5,})", clean)]
        array_sizes += [int(m) for m in re.findall(r"range\(\s*(\d{5,})\s*\)", clean)]
        io = bool(re.search(r"\b(input|print)\s*\(", clean))
        long_long = False  # Python int 无限精度，不存在 long long 溢出概念
        sort_used = bool(re.search(r"\b(sorted|sort)\s*\(", clean))
        memset = bool(re.search(r"\[\s*0\s*\]\s*\*", clean))
        lower = bool(re.search(r"\bbisect\b", clean))
        mod_extra = False
    elif lang == "java":
        array_sizes = [int(m) for m in re.findall(r"new\s+[\w<>\[\]]+\s*\[\s*(\d{5,})\s*\]", clean)]
        io = bool(re.search(r"\bScanner\b|System\s*\.\s*out\b", clean))
        long_long = bool(re.search(r"\blong\b", clean))
        sort_used = bool(re.search(r"Arrays\.sort|Collections\.sort", clean))
        memset = bool(re.search(r"Arrays\.fill", clean))
        lower = bool(re.search(r"binarySearch", clean))
        mod_extra = False
    else:  # cpp（保持与历史版本一致）
        array_sizes = [int(m) for m in re.findall(r"\[\s*(\d{5,})\s*\]", clean)]
        io = bool(re.search(r"\b(cin|cout)\b", clean))
        long_long = bool(re.search(r"\blong\s+long\b|\bll\b", clean))
        sort_used = "sort(" in clean
        memset = "memset" in clean
        lower = "lower_bound" in clean or "upper_bound" in clean
        mod_extra = bool(re.search(r"#define\s+int\s+long\s+long", clean))

    features = {
        "language": lang,
        "lines": len(lines),
        "chars": len(raw),
        "verdict": (verdict or "WA").upper(),
        "loop_nesting": nested,
        "brace_depth": max_depth,
        "recursion": recursive,
        "recursion_func": func_name,
        # 统一保留原有键；语言专属概念在不适用的语言上为 False
        "has_cin_cout": io if lang == "cpp" else False,
        "has_input_output": io,
        "sync_with_stdio": ("sync_with_stdio" in clean) if lang == "cpp" else False,
        "scanf_printf": bool(re.search(r"\b(scanf|printf)\b", clean)) if lang == "cpp" else False,
        "uses_endl": ("endl" in clean) if lang == "cpp" else bool(re.search(r"flush\s*=\s*True", clean)),
        "long_long": long_long,
        "define_int_ll": mod_extra,
        "mod_usage": bool(re.search(r"%\s*(1000000007|1e9\s*\+\s*7|998244353|\w+MOD)", clean)),
        "division": bool(re.search(r"[^/\*]/[^/*=]", clean)),
        "abs_int": bool(re.search(r"\babs\s*\(", clean)) and not re.search(r"\bllabs\s*\(|\bfabs\s*\(", clean),
        "sqrt_as_int": bool(re.search(r"\(\s*int\s*\)\s*sqrt|\bint\s+\w+\s*=\s*sqrt", clean)),
        "memset": memset,
        "large_array": array_sizes,
        "sort_used": sort_used,
        "lower_bound": lower,
        "dp_array": bool(re.search(r"\bdp\s*\[", clean)),
        "visited_array": bool(re.search(r"\b(vis|visited)\s*\[", clean)),
    }
    features["risks"] = detect_risks(features, clean)
    return features


def detect_risks(f: dict, code: str) -> list[str]:
    """把特征翻译成可解释的风险点（也是规则诊断的证据来源）。"""
    risks: list[str] = []
    lang = f.get("language", "cpp")
    if f["loop_nesting"] >= 3:
        risks.append(f"循环嵌套 {f['loop_nesting']} 层，复杂度可能达到 O(n^{f['loop_nesting']})")
    if f["recursion"] and f["verdict"] == "RE":
        risks.append(f"递归函数 {f['recursion_func']} 在 RE 场景下需排查栈深度")
    # 流同步是 C++ 专属概念，仅对 cpp 提示
    if lang == "cpp" and f["has_cin_cout"] and not f["sync_with_stdio"]:
        risks.append("使用 cin/cout 但未关闭流同步，大数据量下可能 TLE")
    if f["uses_endl"]:
        risks.append(("频繁使用 endl 会反复 flush" if lang == "cpp" else "频繁强制 flush 输出") + "，可能造成常数过大")
    if f["division"] and f["mod_usage"]:
        risks.append("存在除法/取模，需检查除零与负数取模")
    if f["large_array"]:
        risks.append(f"静态数组最大 {max(f['large_array'])}，注意内存上限")
    # Python 整数无限精度，不适用 long long 溢出提示
    if lang in ("cpp", "java") and not f["long_long"] and f["mod_usage"]:
        risks.append("有取模运算但未见 long long，乘法可能溢出")
    if f["sqrt_as_int"]:
        risks.append("sqrt 转 int 存在精度误差，需 ±1 校验")
    if f["abs_int"] and lang in ("cpp", "java"):
        risks.append("对 long long 使用 abs 可能溢出，应使用 llabs")
    return risks

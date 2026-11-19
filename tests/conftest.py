# -*- coding: utf-8 -*-
"""pytest 全局配置：把 src 加入 sys.path，避免依赖安装即可运行测试。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

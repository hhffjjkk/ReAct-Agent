# -*- coding: utf-8 -*-
"""pytest 配置：保证在项目根目录直接 `uv run pytest tests/` 时可导入 app 包。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

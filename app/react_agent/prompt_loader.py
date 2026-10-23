"""
Prompt 加载工具：支持 prompts/*.md 中的 {{变量}} 占位符。
单独实现的原因：现有 load_prompt 只读取 .prompt，且使用 str.format（会与 JSON 花括号冲突）。
"""
from pathlib import Path
from typing import Any

from app.utils.path_util import PROJECT_ROOT


def load_md_prompt(name: str, **kwargs: Any) -> str:
    """
    读取 prompts/{name}.md 并替换 {{key}} 占位符。
    :param name: 文件名（不带 .md）
    :param kwargs: 占位符键值
    """
    path = PROJECT_ROOT / "prompts" / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt 文件不存在：{path}")
    text = path.read_text(encoding="utf-8")
    for key, value in kwargs.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text

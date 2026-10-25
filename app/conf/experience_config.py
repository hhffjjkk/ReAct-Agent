# 经验记忆（在线）与技能注入（离线沉淀）配置
# 说明：两个能力默认关闭；关闭时 ReAct Prompt 与原有行为保持完全一致。
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


def _to_bool(value, default=False) -> bool:
    """将 .env 字符串转为布尔值，兼容 1/0、True/False、true/false"""
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _to_int(value, default: int) -> int:
    """安全转换为整数，失败返回默认值"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value, default: float) -> float:
    """安全转换为浮点数，失败返回默认值"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class ExperienceMemoryConf:
    # ---- 能力 A：在线经验记忆 ----
    experience_memory_enabled: bool
    experience_memory_topk: int
    experience_memory_threshold: float
    experience_memory_timeout_ms: int
    experience_memory_collection: str

    # ---- 能力 B：离线技能沉淀 / 在线技能注入 ----
    skill_injection_enabled: bool
    skill_dir: str


experience_config = ExperienceMemoryConf(
    experience_memory_enabled=_to_bool(os.getenv("EXPERIENCE_MEMORY_ENABLED"), False),
    experience_memory_topk=_to_int(os.getenv("EXPERIENCE_MEMORY_TOPK"), 2),
    experience_memory_threshold=_to_float(os.getenv("EXPERIENCE_MEMORY_THRESHOLD"), 0.75),
    experience_memory_timeout_ms=_to_int(os.getenv("EXPERIENCE_MEMORY_TIMEOUT_MS"), 300),
    experience_memory_collection=(os.getenv("EXPERIENCE_MEMORY_COLLECTION") or "diagnosis_experience").strip(),
    skill_injection_enabled=_to_bool(os.getenv("SKILL_INJECTION_ENABLED"), False),
    skill_dir=(os.getenv("SKILL_DIR") or "./skills").strip(),
)

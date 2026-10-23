# ReAct 诊断 Agent 配置
# 说明：所有开关均通过 .env 控制，默认开启；AGENT_ENABLED=false 时新增接口统一返回 404，不影响原有链路。
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


@dataclass
class AgentConfig:
    enabled: bool              # 新增 /agent/* 接口总开关
    max_steps: int             # ReAct 最大工具调用步数（硬上限）
    top_k: int                 # search_knowledge_base 默认返回切片数
    trace_collection: str      # 轨迹集合名
    ticket_collection: str     # 历史工单集合名（复用现有 MongoDB）


agent_config = AgentConfig(
    enabled=_to_bool(os.getenv("AGENT_ENABLED"), True),
    max_steps=_to_int(os.getenv("AGENT_MAX_STEPS"), 6),
    top_k=_to_int(os.getenv("AGENT_TOPK"), 5),
    trace_collection=(os.getenv("AGENT_TRACE_COLLECTION") or "agent_traces").strip(),
    ticket_collection=(os.getenv("AGENT_TICKET_COLLECTION") or "ticket_history").strip(),
)

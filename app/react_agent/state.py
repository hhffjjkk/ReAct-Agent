"""
ReAct 诊断 Agent 的图状态定义。
说明：所有字段均为 LangGraph 节点之间流转的数据，节点返回部分字段即可。
"""
from typing import Any, Dict, List, TypedDict


class AgentGraphState(TypedDict, total=False):
    # 基础信息
    session_id: str
    query: str
    is_stream: bool
    start_ts: float

    # 意图分流
    route: str                 # single_rag | react
    route_reason: str
    history: List[Dict[str, Any]]

    # ReAct 控制
    step: int                  # 已完成的工具调用步数
    max_steps: int
    max_steps_reached: bool
    thought: str
    action: str                # 工具名 或 final
    action_input: Dict[str, Any]
    observation: Dict[str, Any]
    observation_text: str
    scratchpad: List[Dict[str, Any]]   # 每一步的 {step, thought, action, action_input, observation, timestamp}
    tool_results: List[Dict[str, Any]]

    # 结果
    final_answer: str
    answer: str
    transfer_to_human: bool
    error: str
    latency_ms: int

    # 经验记忆 / 技能注入（能力 A / B）
    experience_memory_enabled: bool
    experience_hit: bool
    experience_injected: int
    experience_latency_ms: int
    experience_ids: List[str]
    skill_injection_enabled: bool
    skill_injected: bool
    skill_name: str
    skill_block: str
    experience_block: str


def create_agent_default_state(**overrides) -> AgentGraphState:
    """创建 Agent 图默认状态"""
    state: AgentGraphState = {
        "session_id": "",
        "query": "",
        "is_stream": True,
        "start_ts": 0.0,
        "route": "",
        "route_reason": "",
        "history": [],
        "step": 0,
        "max_steps": 6,
        "max_steps_reached": False,
        "thought": "",
        "action": "",
        "action_input": {},
        "observation": {},
        "observation_text": "",
        "scratchpad": [],
        "tool_results": [],
        "final_answer": "",
        "answer": "",
        "transfer_to_human": False,
        "error": "",
        "latency_ms": 0,
        # 经验记忆 / 技能注入
        "experience_memory_enabled": False,
        "experience_hit": False,
        "experience_injected": 0,
        "experience_latency_ms": 0,
        "experience_ids": [],
        "skill_injection_enabled": False,
        "skill_injected": False,
        "skill_name": "",
        "skill_block": "",
        "experience_block": "",
    }
    state.update(overrides)
    return state

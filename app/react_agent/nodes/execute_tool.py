"""
工具执行节点：按 Action 分发到四个工具之一。
工具不存在 / 参数非法时返回明确的错误 Observation，由 LLM 自行纠正。
"""
import time

from app.core.logger import logger
from app.react_agent.tools import execute_tool


def node_execute_tool(state):
    action = state.get("action", "")
    action_input = state.get("action_input") or {}
    start = time.time()
    result = execute_tool(action, action_input)
    logger.info(f"[AgentTool] step={state.get('step', 0)} action={action} "
                f"cost_ms={int((time.time() - start) * 1000)} hit={result.get('hit')}")
    return {"observation": result}

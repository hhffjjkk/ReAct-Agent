"""
观察节点：把工具结果回填为 Observation，追加到 scratchpad，并推送 SSE。
"""
import json
import time

from app.react_agent.agent_sse import push_agent_event


def node_observe(state):
    step = int(state.get("step", 0) or 0) + 1
    observation = state.get("observation") or {}
    observation_text = json.dumps(observation, ensure_ascii=False)
    if len(observation_text) > 4000:
        observation_text = observation_text[:4000] + "...(截断)"

    entry = {
        "step": step,
        "thought": state.get("thought", ""),
        "action": state.get("action", ""),
        "action_input": state.get("action_input") or {},
        "observation": observation,
        "timestamp": time.time(),
    }
    scratchpad = list(state.get("scratchpad") or []) + [entry]
    tool_results = list(state.get("tool_results") or []) + [{
        "step": step,
        "tool": state.get("action", ""),
        "action_input": state.get("action_input") or {},
        "result": observation,
    }]

    push_agent_event(state["session_id"], "observation", {
        "step": step,
        "action": state.get("action", ""),
        "observation": observation,
    })
    return {
        "step": step,
        "observation_text": observation_text,
        "scratchpad": scratchpad,
        "tool_results": tool_results,
    }

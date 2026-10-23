"""
答案生成 / 收尾节点：
  - 输出 Final Answer（若模型已给出则直接使用）
  - 写入 agent_traces（每步轨迹 + 汇总字段）
  - 通过 SSE 推送 final / done
"""
import time
from typing import Any, Dict, List

from app.core.logger import logger
from app.react_agent.agent_sse import push_agent_event
from app.react_agent.trace_store import save_agent_trace


def _build_trace(state) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = state.get("scratchpad") or []
    start_ts = float(state.get("start_ts", 0) or 0)
    latency_ms = int(state.get("latency_ms", 0) or 0)
    if start_ts:
        latency_ms = max(latency_ms, int((time.time() - start_ts) * 1000))
    answer = state.get("final_answer") or state.get("answer") or ""
    return {
        "session_id": state.get("session_id", ""),
        "query": state.get("query", ""),
        "route": state.get("route", ""),
        "route_reason": state.get("route_reason", ""),
        "status": "completed",
        "step_count": len(steps),
        "max_steps": int(state.get("max_steps", 6) or 6),
        "max_steps_reached": bool(state.get("max_steps_reached")),
        "transfer_to_human": bool(state.get("transfer_to_human")),
        "final_answer": answer,
        "answer_len": len(answer),
        "latency_ms": latency_ms,
        "steps": steps,
        "error": state.get("error", "") or "",
        # 明确标注：工具数据为模拟数据
        "mock_data": True,
        "mock_note": "check_device_status / check_user_permission / query_ticket_history 使用模拟数据，生产环境需替换为真实 CMDB/ITSM/IAM 接口。",
    }


def node_generate_answer(state):
    session_id = state["session_id"]
    answer = state.get("final_answer") or state.get("answer") or ""
    if not answer:
        answer = "未能定位到明确根因，建议转人工工单，由 IT 运维协助排查。"
        state["transfer_to_human"] = True
    state["final_answer"] = answer
    state["answer"] = answer

    trace = _build_trace(state)
    state["latency_ms"] = trace["latency_ms"]
    save_agent_trace(trace)

    logger.info(f"[AgentAnswer] 完成：session_id={session_id}, route={trace['route']}, "
                f"steps={trace['step_count']}, max_steps_reached={trace['max_steps_reached']}, "
                f"transfer={trace['transfer_to_human']}, latency_ms={trace['latency_ms']}")

    push_agent_event(session_id, "final", {
        "answer": answer,
        "route": trace["route"],
        "step_count": trace["step_count"],
        "max_steps_reached": trace["max_steps_reached"],
        "transfer_to_human": trace["transfer_to_human"],
        "latency_ms": trace["latency_ms"],
    })
    push_agent_event(session_id, "done", {"session_id": session_id})
    push_agent_event(session_id, "__close__", {})

    return {"answer": answer, "final_answer": answer, "latency_ms": trace["latency_ms"]}

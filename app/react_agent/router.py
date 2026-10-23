"""
ReAct 诊断 Agent 接口（挂载到查询服务 :8001）。

接口：
  POST /agent/query                发起诊断，返回 session_id
  GET  /agent/stream/{session_id}  SSE 推送 Thought / Action / Observation / Final Answer
  GET  /agent/history/{session_id} 返回完整 ReAct 轨迹（来自 agent_traces）
  GET  /agent/metrics              监控聚合指标
  GET  /agent/monitor.html         监控页面

所有接口受 AGENT_ENABLED 控制；关闭时统一返回 404，不影响原有接口。
"""
import time
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.core.logger import logger
from app.react_agent.agent_sse import (
    agent_sse_generator, create_agent_queue, has_agent_queue, push_agent_event, remove_agent_queue,
)
from app.react_agent.config import agent_config
from app.react_agent.graph import agent_app
from app.react_agent.metrics import build_agent_metrics
from app.react_agent.state import create_agent_default_state
from app.react_agent.trace_store import get_agent_trace, save_agent_trace
from app.utils.path_util import PROJECT_ROOT

agent_router = APIRouter(prefix="/agent", tags=["ReAct Agent"])


def _ensure_enabled() -> None:
    """总开关关闭时返回 404（不影响原有接口）"""
    if not agent_config.enabled:
        raise HTTPException(status_code=404, detail="ReAct Agent 未启用（AGENT_ENABLED=false）")


class AgentQueryRequest(BaseModel):
    query: str = Field(..., title="用户问题，必须传递")
    session_id: Optional[str] = Field(None, title="会话 id，可不传，后台自动生成")


def run_agent_graph(query: str, session_id: str) -> None:
    """后台执行 ReAct 图；异常时兜底写轨迹并推送 error"""
    state = create_agent_default_state(
        session_id=session_id,
        query=query,
        is_stream=True,
        start_ts=time.time(),
        max_steps=agent_config.max_steps,
    )
    try:
        agent_app.invoke(state)
    except Exception as e:
        logger.exception(f"[AgentAPI] 图执行失败：session_id={session_id}, {e}")
        push_agent_event(session_id, "error", {"error": str(e)})
        push_agent_event(session_id, "__close__", {})
        try:
            save_agent_trace({
                "session_id": session_id,
                "query": query,
                "route": "unknown",
                "status": "failed",
                "step_count": 0,
                "max_steps": agent_config.max_steps,
                "max_steps_reached": False,
                "transfer_to_human": True,
                "final_answer": "诊断流程异常终止，建议转人工工单。",
                "latency_ms": int((time.time() - state.get("start_ts", time.time())) * 1000),
                "steps": [],
                "error": f"graph_failed: {e}",
                "mock_data": True,
            })
        except Exception:
            pass


@agent_router.post("/query")
async def agent_query(req: AgentQueryRequest, background_tasks: BackgroundTasks):
    """发起 ReAct 诊断（异步执行，轨迹通过 /agent/stream 推送）"""
    _ensure_enabled()
    session_id = req.session_id or str(uuid.uuid4())
    create_agent_queue(session_id)
    background_tasks.add_task(run_agent_graph, req.query, session_id)
    logger.info(f"[AgentAPI] 已发起诊断：session_id={session_id}, query={req.query!r}")
    return {
        "session_id": session_id,
        "message": "诊断已启动，请通过 /agent/stream/{session_id} 查看 ReAct 轨迹",
    }


@agent_router.get("/stream/{session_id}")
async def agent_stream(session_id: str, request: Request):
    """SSE 流式返回 ReAct 轨迹"""
    _ensure_enabled()
    if not has_agent_queue(session_id):
        remove_agent_queue(session_id)
        raise HTTPException(status_code=404, detail="会话不存在或已结束，可通过 /agent/history/{session_id} 查看轨迹")
    return StreamingResponse(
        agent_sse_generator(session_id, request),
        media_type="text/event-stream",
    )


@agent_router.get("/history/{session_id}")
async def agent_history(session_id: str):
    """返回会话的完整 ReAct 轨迹"""
    _ensure_enabled()
    trace = get_agent_trace(session_id)
    if not trace:
        raise HTTPException(status_code=404, detail="未找到该会话的 ReAct 轨迹")
    return trace


@agent_router.get("/metrics")
async def agent_metrics(days: int = 14):
    """监控聚合指标（供 agent_monitor.html 使用）"""
    _ensure_enabled()
    return build_agent_metrics(days=days)


@agent_router.get("/monitor.html")
async def agent_monitor_page():
    """ReAct 诊断 Agent 监控页"""
    _ensure_enabled()
    page = PROJECT_ROOT / "app" / "query_process" / "page" / "agent_monitor.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="agent_monitor.html 不存在")
    return FileResponse(page)

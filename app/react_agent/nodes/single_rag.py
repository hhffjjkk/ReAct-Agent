"""
单轮 RAG 分支：复用现有查询图的全部节点（不改原图拓扑、不改原节点逻辑）。
说明：为便于 Agent 链路与原有 SSE 通道隔离，这里以 is_stream=False 方式内联调用原图。
"""
import time

from app.core.logger import logger
from app.react_agent.agent_sse import push_agent_event


def node_single_rag(state):
    """调用原单轮 RAG 图，产出 final_answer"""
    session_id = state["session_id"]
    query = state.get("query", "")
    start = time.time()
    try:
        from app.query_process.agent.main_graph import query_app
        from app.query_process.agent.state import create_query_default_state

        inner_state = create_query_default_state(
            session_id=session_id,
            original_query=query,
            is_stream=False,
            query_start_ts=start,
        )
        result = query_app.invoke(inner_state)
        answer = (result or {}).get("answer", "") or "未检索到相关内容，请补充打印机型号或故障现象。"
        latency_ms = int((time.time() - start) * 1000)
        logger.info(f"[AgentSingleRAG] 完成：session_id={session_id}, latency_ms={latency_ms}")
        push_agent_event(session_id, "thought", {
            "step": 0,
            "thought": "该问题属于事实型，直接复用单轮 RAG 知识库检索链路。",
        })
        return {"final_answer": answer, "answer": answer, "latency_ms": latency_ms}
    except Exception as e:
        logger.exception(f"[AgentSingleRAG] 执行失败：{e}")
        return {
            "final_answer": "知识库检索链路暂时不可用，建议稍后重试或转人工工单。",
            "error": f"single_rag_failed: {e}",
            "latency_ms": int((time.time() - start) * 1000),
        }

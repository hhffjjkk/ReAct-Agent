"""
能力 A：在线经验记忆。

职责：
  - 在 ReAct 会话结束后，将“成功诊断”异步写入 diagnosis_experience
  - 在 react_plan 开始前，按 query embedding 检索 top-k 成功经验（带超时保护）
  - 将经验渲染为 prompt 中的【参考经验】块

成功定义（对齐任务文档）：
  max_steps_reached == false 且 retrieval_feedback 中存在该 session_id 的正确反馈。
  最大步数触发的会话可选写入 success=false，用于后续分析，不参与在线检索。
"""
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Dict, List, Optional, Tuple

from app.conf.experience_config import experience_config
from app.core.logger import logger

# 检索超时用独立线程池：超时后不阻塞主流程
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="exp-memory")
_BG_LOOP: Optional[asyncio.AbstractEventLoop] = None
_BG_LOCK = threading.Lock()


# ===================== 反馈查询 =====================
def has_positive_feedback(session_id: str) -> bool:
    """查询 retrieval_feedback 中是否存在该 session 的“正确”反馈"""
    if not session_id:
        return False
    try:
        from app.clients.mongo_monitor_utils import get_monitor_tool
        tool = get_monitor_tool()
        doc = tool.retrieval_feedback.find_one({"session_id": session_id, "correct": True})
        return doc is not None
    except Exception as e:
        logger.warning(f"[ExperienceMemory] 查询反馈失败（按无反馈处理）：{e}")
        return False


# ===================== 写入 =====================
def _tool_sequence(trace: Dict[str, Any]) -> str:
    actions = []
    for step in trace.get("steps", []) or []:
        action = (step.get("action") or "").strip()
        if action and action != "final":
            actions.append(action)
    return "→".join(actions)


def embed_query(query: str) -> Optional[List[float]]:
    """BGE-M3 生成 query 稠密向量（1024 维）；失败返回 None"""
    try:
        from app.lm.embedding_utils import generate_embeddings
        embeddings = generate_embeddings([query])
        return embeddings["dense"][0]
    except Exception as e:
        logger.warning(f"[ExperienceMemory] query 向量生成失败：{e}")
        return None


def write_experience_sync(trace: Dict[str, Any]) -> bool:
    """同步写入一条经验（供异步封装/脚本/测试调用）"""
    if not experience_config.experience_memory_enabled:
        return False
    if trace.get("route") != "react":
        return False
    sequence = _tool_sequence(trace)
    if not sequence:
        return False

    max_steps_reached = bool(trace.get("max_steps_reached"))
    session_id = trace.get("session_id", "")
    if max_steps_reached:
        success = False
    elif has_positive_feedback(session_id):
        success = True
    else:
        # 没有人工确认正确的反馈：不写入 success=true
        logger.info(f"[ExperienceMemory] 会话无正确反馈，跳过经验写入：session_id={session_id}")
        return False

    query_text = (trace.get("query") or "").strip()
    embedding = embed_query(query_text)
    if not embedding:
        return False

    created_at = int(float(trace.get("created_at") or time.time()))
    experience = {
        "exp_id": f"{session_id}_{created_at}",
        "query_text": query_text,
        "query_embedding": embedding,
        "tool_sequence": sequence,
        "final_answer": (trace.get("final_answer") or "")[:500],
        "success": success,
        "source_dept": trace.get("source_dept", "") or "",
        "create_time": int(time.time()),
    }
    from app.clients.experience_milvus_utils import insert_experience
    ok = insert_experience(experience)
    if ok:
        logger.info(f"[ExperienceMemory] 经验写入成功：session_id={session_id}, success={success}, "
                    f"sequence={sequence}")
    return ok


async def write_experience_from_trace(trace: Dict[str, Any]) -> bool:
    """异步写入（阻塞的 Milvus/embedding 调用放到线程中）"""
    return await asyncio.to_thread(write_experience_sync, trace)


def _get_bg_loop() -> asyncio.AbstractEventLoop:
    """获取后台事件循环（没有运行中的 loop 时使用）"""
    global _BG_LOOP
    with _BG_LOCK:
        if _BG_LOOP is None or _BG_LOOP.is_closed():
            _BG_LOOP = asyncio.new_event_loop()
            thread = threading.Thread(
                target=_BG_LOOP.run_forever,
                name="experience-memory-loop",
                daemon=True,
            )
            thread.start()
        return _BG_LOOP


def schedule_experience_write(trace: Dict[str, Any]) -> None:
    """
    异步调度经验写入，不阻塞 SSE 结束。
    优先使用当前事件循环的 asyncio.create_task；无 loop 时投递到后台 loop。
    """
    if not experience_config.experience_memory_enabled:
        return
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(write_experience_from_trace(trace))
    except RuntimeError:
        asyncio.run_coroutine_threadsafe(
            write_experience_from_trace(trace), _get_bg_loop()
        )


# ===================== 检索 =====================
def _retrieve_sync(query: str) -> List[Dict[str, Any]]:
    embedding = embed_query(query)
    if not embedding:
        return []
    from app.clients.experience_milvus_utils import search_experiences
    return search_experiences(
        query_embedding=embedding,
        topk=experience_config.experience_memory_topk,
        threshold=experience_config.experience_memory_threshold,
    )


def retrieve_experiences(query: str) -> Tuple[List[Dict[str, Any]], int, bool]:
    """
    带超时保护的经验检索。
    :return: (经验列表, 检索耗时毫秒, 是否超时)
    """
    if not experience_config.experience_memory_enabled:
        return [], 0, False

    start = time.perf_counter()
    future = _EXECUTOR.submit(_retrieve_sync, query)
    try:
        items = future.result(timeout=max(1, experience_config.experience_memory_timeout_ms) / 1000.0)
        latency = int((time.perf_counter() - start) * 1000)
        logger.info(f"[ExperienceMemory] 经验检索完成：命中 {len(items)} 条，耗时 {latency}ms")
        return items, latency, False
    except FutureTimeout:
        latency = int((time.perf_counter() - start) * 1000)
        logger.warning(f"[ExperienceMemory] 经验检索超时（>{experience_config.experience_memory_timeout_ms}ms），跳过注入")
        return [], latency, True
    except Exception as e:
        latency = int((time.perf_counter() - start) * 1000)
        logger.warning(f"[ExperienceMemory] 经验检索失败（跳过注入）：{e}")
        return [], latency, False


# ===================== Prompt 渲染 =====================
def build_experience_block(experiences: List[Dict[str, Any]]) -> str:
    """把检索到的成功经验渲染为【参考经验】块；无有效经验返回空串"""
    valid = [x for x in experiences if x.get("success") is True]
    if not valid:
        return ""
    lines = ["【参考经验】（来自历史成功诊断，仅供参考，如与当前情况不符请忽略）", ""]
    for idx, exp in enumerate(valid[:experience_config.experience_memory_topk], 1):
        score = float(exp.get("score") or 0.0)
        query_text = (exp.get("query_text") or "")[:80]
        answer = (exp.get("final_answer") or "")[:200]
        lines.append(
            f"- 案例 {idx}（相似度 {score:.2f}）：用户报\"{query_text}\"，"
            f"工具序列 {exp.get('tool_sequence', '')}，结论摘要：{answer}"
        )
    return "\n".join(lines)

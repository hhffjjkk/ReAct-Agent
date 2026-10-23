"""
Agent 轨迹 SSE 通道。
与原有 sse_utils 分开维护，避免影响原 /stream 的事件语义。
事件类型：route / thought / action / observation / final / done / error
"""
import asyncio
import json
import queue
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import Request

_agent_streams: Dict[str, "queue.Queue"] = {}


def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def create_agent_queue(session_id: str) -> "queue.Queue":
    q = queue.Queue()
    _agent_streams[session_id] = q
    return q


def get_agent_queue(session_id: str) -> Optional["queue.Queue"]:
    return _agent_streams.get(session_id)


def remove_agent_queue(session_id: str) -> None:
    _agent_streams.pop(session_id, None)


def push_agent_event(session_id: str, event: str, data: Dict[str, Any]) -> None:
    """推送轨迹事件；队列不存在时静默忽略（例如客户端尚未连接）"""
    q = _agent_streams.get(session_id)
    if q is None:
        return
    q.put({"event": event, "data": data})


def has_agent_queue(session_id: str) -> bool:
    return session_id in _agent_streams


async def agent_sse_generator(session_id: str, request: Request) -> AsyncGenerator[str, None]:
    """SSE 生成器：连接建立后按顺序推送轨迹事件"""
    q = _agent_streams.get(session_id)
    if q is None:
        return
    loop = asyncio.get_running_loop()
    try:
        yield _sse_pack("ready", {"session_id": session_id})
        while True:
            if await request.is_disconnected():
                break
            try:
                msg = await loop.run_in_executor(None, q.get, True, 1.0)
            except queue.Empty:
                continue

            event = msg.get("event")
            data = msg.get("data")
            if event == "__close__":
                break
            yield _sse_pack(event, data)
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        return
    except Exception as e:
        print(f"[AgentSSE] 生成器异常：{e}")
    finally:
        remove_agent_queue(session_id)

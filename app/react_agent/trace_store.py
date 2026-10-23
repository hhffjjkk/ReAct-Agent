"""
ReAct 轨迹存储：写入 / 读取 MongoDB 的 agent_traces 集合（复用现有 Mongo 连接与库）。
设计：一条会话一条文档，steps 内保存每一步完整轨迹，便于 /agent/history 与监控聚合。
"""
import time
from typing import Any, Dict, List, Optional

from pymongo import ASCENDING, DESCENDING

from app.core.logger import logger
from app.react_agent.config import agent_config


def _collection():
    """复用现有 Mongo 连接（不新增基础设施）"""
    from app.clients.mongo_history_utils import get_history_mongo_tool
    tool = get_history_mongo_tool()
    coll = tool.db[agent_config.trace_collection]
    # create_index 幂等
    coll.create_index([("session_id", ASCENDING)], unique=True)
    coll.create_index([("created_at", DESCENDING)])
    return coll


def save_agent_trace(trace: Dict[str, Any]) -> bool:
    """按 session_id upsert 一条轨迹；任何异常只记录日志，不影响主流程"""
    try:
        session_id = trace.get("session_id")
        if not session_id:
            return False
        coll = _collection()
        doc = dict(trace)
        doc.setdefault("created_at", time.time())
        doc["updated_at"] = time.time()
        coll.update_one({"session_id": session_id}, {"$set": doc}, upsert=True)
        logger.info(f"[AgentTrace] 轨迹已写入：session_id={session_id}, route={doc.get('route')}, "
                    f"steps={doc.get('step_count')}")
        return True
    except Exception as e:
        logger.error(f"[AgentTrace] 写入轨迹失败：{e}")
        return False


def get_agent_trace(session_id: str) -> Optional[Dict[str, Any]]:
    """按 session_id 读取完整轨迹（去掉不可序列化的 _id）"""
    try:
        doc = _collection().find_one({"session_id": session_id})
        if not doc:
            return None
        doc.pop("_id", None)
        return doc
    except Exception as e:
        logger.error(f"[AgentTrace] 读取轨迹失败：{e}")
        return None


def list_agent_traces(limit: int = 500, since_ts: float = 0) -> List[Dict[str, Any]]:
    """按时间倒序读取轨迹列表"""
    try:
        query = {}
        if since_ts:
            query["created_at"] = {"$gte": since_ts}
        cursor = _collection().find(query).sort("created_at", DESCENDING).limit(int(limit))
        docs = []
        for doc in cursor:
            doc.pop("_id", None)
            docs.append(doc)
        return docs
    except Exception as e:
        logger.error(f"[AgentTrace] 列表读取失败：{e}")
        return []

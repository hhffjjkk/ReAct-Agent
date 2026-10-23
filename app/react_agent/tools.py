"""
ReAct 诊断 Agent 的四个工具。

约束（见任务文档）：
  - 工具数量固定为 4 个：search_knowledge_base / query_ticket_history /
    check_device_status / check_user_permission
  - 每个工具返回结构化 dict（不返回整篇文档），并带 hit 字段
  - description 描述“何时使用”，作为 LLM 选择工具的依据
  - 工具内部异常统一转为错误结果，交由 LLM 在 Observation 中自行纠正
"""
import re
from typing import Any, Dict, List, Optional

from app.core.logger import logger
from app.react_agent.config import agent_config
from app.react_agent.mock_source import mock_check_device_status, mock_check_user_permission
from app.utils.doc_meta_utils import now_epoch, epoch_to_date_str, compute_timeliness_factor

# ===================== 工具清单（供 Prompt 与校验共用） =====================
TOOL_DEFINITIONS = [
    {
        "name": "search_knowledge_base",
        "description": (
            "当用户询问打印机安装、驱动配置、报错码含义或标准处理步骤，"
            "需要从企业知识库/手册中检索权威资料时调用。"
        ),
        "parameters": {
            "query": "字符串，必填，检索关键词（尽量包含打印机型号、错误码或故障现象）"
        },
    },
    {
        "name": "query_ticket_history",
        "description": (
            "当用户描述具体故障现象（如打印空白、卡纸、任务卡住），"
            "希望参考同类历史工单的处理方案时调用。"
        ),
        "parameters": {
            "symptom": "字符串，必填，故障现象描述（可用用户原话）"
        },
    },
    {
        "name": "check_device_status",
        "description": (
            "当用户报告打印机无响应、任务卡住或出现错误码时调用，"
            "用于确认设备是否在线及队列状态。"
        ),
        "parameters": {
            "device_id": "字符串，可选，设备编号（如 PRN-002）",
            "printer_name": "字符串，可选，打印机名称（如 3F-财务-奔图-P3302）",
        },
    },
    {
        "name": "check_user_permission",
        "description": (
            "当怀疑用户没有该打印机的使用权限，或客户端驱动版本与设备要求不一致时调用。"
        ),
        "parameters": {
            "user_id": "字符串，必填，用户工号（如 U1001）",
            "device_id": "字符串，必填，设备编号（如 PRN-001）",
        },
    },
]


def render_tools_for_prompt() -> str:
    """把工具清单渲染为 Prompt 文本"""
    lines: List[str] = []
    for tool in TOOL_DEFINITIONS:
        params = "；".join(f"{k}: {v}" for k, v in tool["parameters"].items())
        lines.append(f"- {tool['name']}\n  何时使用：{tool['description']}\n  参数：{params}")
    return "\n".join(lines)


# ===================== 工具 1：知识库检索 =====================
def search_knowledge_base(query: str) -> Dict[str, Any]:
    """
    复用现有 Milvus 混合检索 + BGE-M3 + Reranker，
    并继承现有版本/时效过滤（过期内容不召回）。
    """
    query = (query or "").strip()
    if not query:
        return {"hit": False, "error": "参数 query 不能为空", "results": []}

    try:
        # 延迟导入重型模型依赖，避免影响原有服务启动
        from app.lm.embedding_utils import generate_embeddings
        from app.lm.reranker_utils import get_reranker_model
        from app.clients.milvus_utils import (
            get_milvus_client, build_chunk_filter_expr,
            create_hybrid_search_requests, hybrid_search,
        )
        from app.conf.milvus_config import milvus_config
        from app.conf.timeliness_config import timeliness_config

        recall_limit = max(10, agent_config.top_k * 2)
        embeddings = generate_embeddings([query])
        # 关键：复用现有过滤表达式（is_latest + 时效硬过滤 + 生效下界）
        expr = build_chunk_filter_expr([], now_ts=now_epoch())
        reqs = create_hybrid_search_requests(
            dense_vector=embeddings["dense"][0],
            sparse_vector=embeddings["sparse"][0],
            expr=expr,
            limit=recall_limit,
        )
        client = get_milvus_client()
        if client is None:
            return {"hit": False, "error": "Milvus 连接不可用", "results": []}
        # 知识库尚未导入文档时，直接返回“未命中”，避免无意义的 Milvus 异常日志
        if hasattr(client, "has_collection") and not client.has_collection(milvus_config.chunks_collection):
            return {
                "hit": False,
                "query": query,
                "count": 0,
                "results": [],
                "message": "知识库集合尚不存在（未导入文档），请勿编造答案。",
            }

        resp = hybrid_search(
            client=client,
            collection_name=milvus_config.chunks_collection,
            reqs=reqs,
            ranker_weights=(0.9, 0.1),
            norm_score=True,
            limit=recall_limit,
            output_fields=[
                "chunk_id", "content", "file_title", "title",
                "publish_date", "expire_date", "source_dept",
            ],
        )
        hits = resp[0] if resp else []
        docs: List[Dict[str, Any]] = []
        for hit in hits:
            entity = hit.get("entity") or {}
            content = entity.get("content") or ""
            if not content:
                continue
            docs.append({
                "chunk_id": entity.get("chunk_id", ""),
                "content": content,
                "source_doc": entity.get("file_title") or entity.get("title") or "未知来源",
                "publish_date_ts": int(entity.get("publish_date") or 0),
                "expire_date_ts": int(entity.get("expire_date") or 0),
                "timeliness_status": entity.get("timeliness_status", ""),
                "source_dept": entity.get("source_dept", ""),
            })

        if not docs:
            return {
                "hit": False,
                "query": query,
                "count": 0,
                "results": [],
                "message": "知识库未检索到有效资料（可能不相关或已过期），请勿编造答案。",
            }

        # Reranker 精排
        reranker = get_reranker_model()
        pairs = [[query, doc["content"]] for doc in docs]
        scores = reranker.compute_score(pairs, normalize=True)
        if isinstance(scores, (int, float)):
            scores = [scores]
        now_ts = now_epoch()
        for score, doc in zip(scores, docs):
            factor = 1.0
            if timeliness_config.timeliness_enabled:
                factor = compute_timeliness_factor(
                    publish_date=doc["publish_date_ts"],
                    now_ts=now_ts,
                    half_life_days=timeliness_config.timeliness_half_life_days,
                    min_factor=timeliness_config.timeliness_min_factor,
                )
                # 软策略下过期内容额外降权（硬策略下已在 Milvus 过滤）
                if (timeliness_config.timeliness_expired_policy == "soft"
                        and doc["expire_date_ts"] and doc["expire_date_ts"] <= now_ts):
                    factor *= 0.3
            doc["score"] = round(float(score) * factor, 4)
        docs.sort(key=lambda x: x["score"], reverse=True)

        results = []
        for doc in docs[:agent_config.top_k]:
            results.append({
                "content": doc["content"],
                "source_doc": doc["source_doc"],
                "publish_date": epoch_to_date_str(doc["publish_date_ts"], default=""),
                "expire_date": epoch_to_date_str(doc["expire_date_ts"], default=""),
                "score": doc["score"],
            })
        logger.info(f"[AgentTool] search_knowledge_base 命中 {len(results)} 条：{query}")
        return {"hit": True, "query": query, "count": len(results), "results": results}

    except Exception as e:
        logger.exception(f"[AgentTool] search_knowledge_base 执行失败：{e}")
        return {"hit": False, "error": f"知识库检索失败：{e}", "query": query, "results": []}


# ===================== 工具 2：历史工单检索 =====================
# ==================== 模拟数据源（生产环境需替换为真实 ITSM 工单接口） ====================
MOCK_TICKETS = [
    {
        "ticket_id": "TK-2026-1001",
        "symptom": "打印机提示错误码 E3，打印任务卡在队列里无法输出",
        "resolution": "重启打印机电源，清理打印队列中卡住的任务，检查纸张与硒鼓安装后再试。",
        "resolved_at": "2026-09-18",
    },
    {
        "ticket_id": "TK-2026-1002",
        "symptom": "打印机无响应、处于脱机状态，电脑显示打印失败",
        "resolution": "检查打印机电源与 USB/网络连接，在系统中取消“脱机使用打印机”，重启 Print Spooler 服务。",
        "resolved_at": "2026-09-25",
    },
    {
        "ticket_id": "TK-2026-1003",
        "symptom": "打印出来是空白页或字迹很淡",
        "resolution": "检查墨粉/硒鼓余量，执行打印头清洁，必要时更换硒鼓。",
        "resolved_at": "2026-10-02",
    },
    {
        "ticket_id": "TK-2026-1004",
        "symptom": "驱动安装失败，系统无法识别打印机或提示驱动版本不匹配",
        "resolution": "卸载旧驱动并清理打印队列，从官网下载与设备匹配的驱动版本重新安装，重启电脑。",
        "resolved_at": "2026-10-10",
    },
    {
        "ticket_id": "TK-2026-1005",
        "symptom": "没有权限使用该打印机，打印任务被拒绝",
        "resolution": "在 IT 服务台申请加入对应打印机组，权限同步后重新添加打印机。",
        "resolved_at": "2026-10-15",
    },
]
# ==================== 模拟数据源结束 ====================


def _ticket_collection():
    """复用现有 Mongo 连接，不新增基础设施"""
    from app.clients.mongo_history_utils import get_history_mongo_tool
    tool = get_history_mongo_tool()
    return tool.db[agent_config.ticket_collection]


def _seed_mock_tickets_if_empty() -> None:
    """首次调用时写入 5 条模拟工单，便于演示不同分支"""
    coll = _ticket_collection()
    if coll.count_documents({}, limit=1) == 0:
        docs = [dict(t, _mock=True) for t in MOCK_TICKETS]
        coll.insert_many(docs)
        logger.info(f"[AgentTool] 已初始化模拟历史工单：{len(docs)} 条（Mock 数据）")


def _tokens(text: str) -> set:
    """轻量关键词集合：英文/数字按词，中文按二元组（不引入向量化依赖）"""
    text = (text or "").lower()
    tokens = set(re.findall(r"[a-z0-9]+", text))
    chars = re.findall(r"[\u4e00-\u9fff]", text)
    for i in range(len(chars) - 1):
        tokens.add(chars[i] + chars[i + 1])
    return tokens


def query_ticket_history(symptom: str) -> Dict[str, Any]:
    """
    在 MongoDB 中按关键词相似度查询历史工单（不要求向量化）。
    返回 {tickets: [{ticket_id, symptom, resolution, resolved_at}], hit: bool}
    """
    symptom = (symptom or "").strip()
    if not symptom:
        return {"hit": False, "error": "参数 symptom 不能为空", "tickets": []}

    try:
        _seed_mock_tickets_if_empty()
        coll = _ticket_collection()
        query_tokens = _tokens(symptom)
        scored = []
        for ticket in coll.find({}):
            tk = _tokens(ticket.get("symptom", ""))
            score = len(query_tokens & tk)
            if score > 0:
                scored.append((score, ticket))
        scored.sort(key=lambda x: x[0], reverse=True)

        tickets = [
            {
                "ticket_id": t.get("ticket_id", ""),
                "symptom": t.get("symptom", ""),
                "resolution": t.get("resolution", ""),
                "resolved_at": t.get("resolved_at", ""),
            }
            for _, t in scored[:3]
        ]
        logger.info(f"[AgentTool] query_ticket_history 命中 {len(tickets)} 条：{symptom}")
        return {"hit": bool(tickets), "symptom": symptom, "count": len(tickets), "tickets": tickets}

    except Exception as e:
        logger.exception(f"[AgentTool] query_ticket_history 执行失败：{e}")
        return {"hit": False, "error": f"历史工单查询失败：{e}", "tickets": []}


# ===================== 工具 3/4：mock 数据源封装 =====================
def check_device_status(device_id: str = "", printer_name: str = "") -> Dict[str, Any]:
    """确认设备是否在线及队列状态（Mock 数据源，生产环境需替换为真实 CMDB 接口）"""
    result = mock_check_device_status(device_id or None, printer_name or None)
    logger.info(f"[AgentTool] check_device_status 返回：{result}")
    return result


def check_user_permission(user_id: str = "", device_id: str = "") -> Dict[str, Any]:
    """确认用户打印权限与驱动版本（Mock 数据源，生产环境需替换为真实 IAM/ITSM 接口）"""
    result = mock_check_user_permission(user_id or None, device_id or None)
    logger.info(f"[AgentTool] check_user_permission 返回：{result}")
    return result


# ===================== 统一分发 =====================
TOOL_REGISTRY = {
    "search_knowledge_base": search_knowledge_base,
    "query_ticket_history": query_ticket_history,
    "check_device_status": check_device_status,
    "check_user_permission": check_user_permission,
}


def execute_tool(name: str, action_input: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    按工具名分发；工具不存在或参数非法时返回明确错误，供 LLM 在 Observation 中纠正。
    绝不向上抛异常。
    """
    name = (name or "").strip()
    if name not in TOOL_REGISTRY:
        return {
            "hit": False,
            "error": f"工具不存在：{name}。可用工具：{', '.join(TOOL_REGISTRY.keys())}",
        }
    if action_input is None:
        action_input = {}
    if not isinstance(action_input, dict):
        return {"hit": False, "error": "Action Input 必须是 JSON 对象，例如 {\"query\": \"...\"}"}

    fn = TOOL_REGISTRY[name]
    allowed = set(fn.__code__.co_varnames[:fn.__code__.co_argcount])
    kwargs = {k: v for k, v in action_input.items() if k in allowed}
    try:
        return fn(**kwargs)
    except TypeError as e:
        return {"hit": False, "error": f"参数不合法：{e}。请参考工具参数说明重新调用。"}
    except Exception as e:
        logger.exception(f"[AgentTool] 工具 {name} 执行异常：{e}")
        return {"hit": False, "error": f"工具执行异常：{e}"}

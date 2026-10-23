"""
意图分流节点。
策略：强规则优先（保证验收用例稳定），规则无法判断时调用一次轻量 LLM。
无论走哪条路径都会记录 route 日志，并通过 SSE 推送 route 事件。
"""
import json
from typing import Optional

from langchain_core.messages import HumanMessage

from app.core.logger import logger
from app.lm.lm_utils import get_llm_client
from app.react_agent.agent_sse import push_agent_event
from app.react_agent.prompt_loader import load_md_prompt

# 诊断型关键词：出现即走 ReAct
_DIAGNOSTIC_KEYWORDS = [
    "打印不了", "无法打印", "不能打印", "打不出来", "打印失败", "打印异常",
    "无响应", "没反应", "卡住", "卡纸", "报错", "错误码", "脱机", "offline",
    "e3", "e1", "权限", "驱动版本", "队列", "空白页", "乱码", "偏色",
]
# 事实型关键词：出现且无诊断关键词时走单轮 RAG
_FACTUAL_KEYWORDS = [
    "怎么安装", "如何安装", "安装方法", "安装步骤", "怎么使用", "如何使用",
    "使用说明", "说明书", "手册", "是什么", "配置方法", "怎么设置", "要求",
]


def _rule_route(query: str) -> Optional[str]:
    text = (query or "").lower()
    if any(k in text for k in _DIAGNOSTIC_KEYWORDS):
        return "react"
    if any(k in text for k in _FACTUAL_KEYWORDS):
        return "single_rag"
    return None


def _llm_route(query: str) -> str:
    """轻量 LLM 分流；失败时默认 single_rag（事实型优先，避免误触发多步诊断）"""
    try:
        prompt = load_md_prompt("intent_router", query=query)
        client = get_llm_client(json_mode=True)
        resp = client.invoke([HumanMessage(content=prompt)])
        text = (resp.content or "").strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        route = (data.get("route") or "").strip()
        if route in ("single_rag", "react"):
            return route
    except Exception as e:
        logger.warning(f"[AgentIntent] LLM 分流失败，默认 single_rag：{e}")
    return "single_rag"


def node_intent_router(state):
    """分流入参：query；输出：route / route_reason"""
    query = state.get("query", "") or ""
    rule = _rule_route(query)
    if rule:
        route, reason, source = rule, "规则命中关键词", "rule"
    else:
        route, reason, source = _llm_route(query), "LLM 判定", "llm"

    logger.info(f"[AgentIntent] route={route} source={source} query={query!r} reason={reason}")
    push_agent_event(state["session_id"], "route", {"route": route, "reason": reason})
    return {"route": route, "route_reason": reason}

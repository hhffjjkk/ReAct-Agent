"""
Agent 监控指标聚合（供 agent_monitor.html 使用）。
数据来源：
  - agent_traces：ReAct / Agent 分流单轮 RAG 的请求量、延迟、步数、工具命中情况
  - retrieval_logs（原有监控集合）：原有单轮 RAG 链路请求量与平均延迟
"""
import time
from typing import Any, Dict, List

from app.conf.monitor_config import monitor_config
from app.react_agent.config import agent_config
from app.react_agent.trace_store import list_agent_traces


def build_agent_metrics(days: int = 14, recent_limit: int = 50) -> Dict[str, Any]:
    since_ts = time.time() - max(1, days) * 86400
    traces = list_agent_traces(limit=10000, since_ts=since_ts)
    react_traces = [t for t in traces if t.get("route") == "react"]
    single_traces = [t for t in traces if t.get("route") == "single_rag"]

    # ---- ReAct 步数分布 ----
    max_steps = agent_config.max_steps
    step_dist = {str(i): 0 for i in range(1, max_steps + 1)}
    step_dist["max_steps_reached"] = 0
    for t in react_traces:
        if t.get("max_steps_reached"):
            step_dist["max_steps_reached"] += 1
        else:
            key = str(int(t.get("step_count") or 0))
            if key in step_dist:
                step_dist[key] += 1
    step_distribution = [{"bucket": k, "count": v} for k, v in step_dist.items()]

    # ---- 工具调用统计 ----
    tool_stats: Dict[str, Dict[str, Any]] = {}
    for t in react_traces:
        for step in t.get("steps", []) or []:
            name = step.get("action") or ""
            if not name or name == "final":
                continue
            item = tool_stats.setdefault(name, {"name": name, "calls": 0, "hit_false": 0})
            item["calls"] += 1
            obs = step.get("observation") or {}
            if obs and obs.get("hit") is False:
                item["hit_false"] += 1
    tools = sorted(tool_stats.values(), key=lambda x: x["calls"], reverse=True)

    # ---- 转人工率 ----
    transfer_count = sum(1 for t in react_traces if t.get("transfer_to_human"))
    transfer_rate = round(transfer_count / len(react_traces), 4) if react_traces else 0.0

    # ---- 经验记忆 / 技能注入可观测 ----
    experience_hit_count = sum(1 for t in react_traces if t.get("experience_hit"))
    experience_injected_count = sum(int(t.get("experience_injected") or 0) for t in react_traces)
    experience_latencies = [
        int(t.get("experience_latency_ms") or 0)
        for t in react_traces
        if t.get("experience_memory_enabled")
    ]
    experience_hit_rate = (
        round(experience_hit_count / len(react_traces), 4) if react_traces else 0.0
    )
    avg_experience_latency = (
        int(sum(experience_latencies) / len(experience_latencies))
        if experience_latencies else 0
    )
    skill_injected_count = sum(1 for t in react_traces if t.get("skill_injected"))

    # ---- 两条链路请求量与平均延迟对比 ----
    def _avg_latency(items: List[Dict[str, Any]]) -> int:
        values = [int(x.get("latency_ms") or 0) for x in items]
        return int(sum(values) / len(values)) if values else 0

    baseline_logs: List[Dict[str, Any]] = []
    if monitor_config.monitor_enabled:
        try:
            from app.clients.mongo_monitor_utils import get_recent_logs
            baseline_logs = get_recent_logs(limit=10000, since_ts=since_ts)
        except Exception:
            baseline_logs = []

    route_comparison = [
        {
            "route": "单轮 RAG（原有 /query）",
            "count": len(baseline_logs),
            "avg_latency_ms": _avg_latency(baseline_logs),
        },
        {
            "route": "单轮 RAG（Agent 分流）",
            "count": len(single_traces),
            "avg_latency_ms": _avg_latency(single_traces),
        },
        {
            "route": "ReAct 诊断",
            "count": len(react_traces),
            "avg_latency_ms": _avg_latency(react_traces),
        },
    ]

    recent = []
    for t in traces[:recent_limit]:
        recent.append({
            "session_id": t.get("session_id", ""),
            "query": t.get("query", ""),
            "route": t.get("route", ""),
            "route_reason": t.get("route_reason", ""),
            "step_count": t.get("step_count", 0),
            "max_steps_reached": bool(t.get("max_steps_reached")),
            "transfer_to_human": bool(t.get("transfer_to_human")),
            "latency_ms": t.get("latency_ms", 0),
            "created_at": t.get("created_at", 0),
            "final_answer": t.get("final_answer", ""),
        })

    return {
        "window_days": days,
        "generated_at": time.time(),
        "summary": {
            "total": len(traces),
            "react_count": len(react_traces),
            "single_rag_count": len(single_traces),
            "transfer_count": transfer_count,
            "transfer_rate": transfer_rate,
            "avg_latency_react_ms": _avg_latency(react_traces),
            "avg_latency_single_rag_ms": _avg_latency(single_traces),
            "max_steps_count": step_dist["max_steps_reached"],
            "experience_memory_hit_rate": experience_hit_rate,
            "experience_memory_injected_count": experience_injected_count,
            "avg_retrieval_latency_ms": avg_experience_latency,
            "skill_injected_count": skill_injected_count,
        },
        "experience_memory": {
            "enabled": any(t.get("experience_memory_enabled") for t in react_traces),
            "hit_count": experience_hit_count,
            "hit_rate": experience_hit_rate,
            "injected_count": experience_injected_count,
            "avg_retrieval_latency_ms": avg_experience_latency,
        },
        "step_distribution": step_distribution,
        "tool_stats": tools,
        "route_comparison": route_comparison,
        "recent": recent,
    }

"""
能力 B：离线技能沉淀脚本。

逻辑：
  1. 读取最近 N 天 agent_traces（max_steps_reached=false 且 retrieval_feedback 为正确的轨迹）
  2. 按 tool_sequence 聚类，出现次数 >= min-count 的组生成一个 SKILL.md
  3. 默认 dry-run 只打印；--write 写入 skills/ 目录

用法：
  uv run python -m app.tool.build_skills              # 默认 dry-run
  uv run python -m app.tool.build_skills --write      # 写入 skills/
  uv run python -m app.tool.build_skills --days 30 --min-count 3
"""
import argparse
import hashlib
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from app.conf.experience_config import experience_config
from app.core.logger import logger
from app.react_agent.experience_memory import has_positive_feedback
from app.react_agent.trace_store import list_agent_traces
from app.utils.path_util import PROJECT_ROOT

_ERROR_CODE_RE = re.compile(r"[A-Za-z]\d{1,4}")
_KEYWORDS = ("卡纸", "脱机", "不出纸", "无响应", "错误码", "队列", "权限", "驱动", "缺纸", "卡住")

_SKILL_NAME_HINTS = {
    ("check_device_status", "query_ticket_history"): "打印设备状态与历史工单排查",
    ("search_knowledge_base", "query_ticket_history"): "知识库与历史工单联合排查",
    ("check_device_status", "search_knowledge_base"): "设备状态与知识库联合排查",
    ("check_device_status", "search_knowledge_base", "query_ticket_history"): "打印故障综合诊断",
}

_TOOL_STEP_HINTS = {
    "check_device_status": "确认设备在线状态与队列",
    "query_ticket_history": "参考历史工单处理方案",
    "search_knowledge_base": "检索知识库手册与故障处理资料",
    "check_user_permission": "确认打印权限与驱动版本",
}


def _tool_sequence(trace: Dict[str, Any]) -> str:
    actions = []
    for step in trace.get("steps", []) or []:
        action = (step.get("action") or "").strip()
        if action and action != "final":
            actions.append(action)
    return "→".join(actions)


def _skill_name(sequence: str) -> str:
    tools = tuple(sequence.split("→"))
    if tools in _SKILL_NAME_HINTS:
        return _SKILL_NAME_HINTS[tools]
    for combo, name in _SKILL_NAME_HINTS.items():
        if all(t in tools for t in combo):
            return name
    return "诊断技能：" + " → ".join(tools)


def _trigger_text(traces: List[Dict[str, Any]]) -> str:
    codes, keywords = set(), set()
    for trace in traces:
        query = trace.get("query", "") or ""
        codes.update(_ERROR_CODE_RE.findall(query))
        for kw in _KEYWORDS:
            if kw in query:
                keywords.add(kw)
    parts = list(codes) + list(keywords)
    return " / ".join(dict.fromkeys(parts)) if parts else "相近故障描述"


def _judgement_text(traces: List[Dict[str, Any]]) -> str:
    """从最终结论中提炼判断逻辑（取最近一条成功轨迹的结论前 300 字）"""
    latest = sorted(traces, key=lambda x: x.get("created_at") or 0, reverse=True)[0]
    answer = (latest.get("final_answer") or "").strip()
    summary = re.sub(r"\s+", " ", answer)[:300]
    return summary or "参考工具返回结果给出排查结论。"


def _render_skill(sequence: str, traces: List[Dict[str, Any]], date_str: str) -> str:
    tools = sequence.split("→")
    lines = [
        f"# Skill: {_skill_name(sequence)}",
        "## 触发条件",
        _trigger_text(traces),
        "## 工具调用顺序",
    ]
    for idx, tool in enumerate(tools, 1):
        lines.append(f"{idx}. {tool} — {_TOOL_STEP_HINTS.get(tool, '调用诊断工具')}")
    lines += [
        "## 关键判断",
        _judgement_text(traces),
        "## 来源",
        f"基于 {len(traces)} 条成功轨迹，最近更新 {date_str}",
        "",
    ]
    return "\n".join(lines)


def _safe_file_name(name: str, sequence: str) -> str:
    safe = re.sub(r"[\\/:*?\"<>|\s]+", "_", name).strip("_") or "skill"
    digest = hashlib.md5(sequence.encode("utf-8")).hexdigest()[:6]
    return f"{safe}_{digest}.md"


def build_skills(days: int = 30, min_count: int = 3, write: bool = False) -> List[Dict[str, Any]]:
    since_ts = time.time() - max(1, days) * 86400
    traces = list_agent_traces(limit=10000, since_ts=since_ts)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        if trace.get("route") != "react" or trace.get("max_steps_reached"):
            continue
        sequence = _tool_sequence(trace)
        if not sequence:
            continue
        if has_positive_feedback(trace.get("session_id", "")):
            grouped[sequence].append(trace)

    date_str = datetime.now().strftime("%Y-%m-%d")
    skills = []
    for sequence, items in sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True):
        if len(items) < min_count:
            continue
        name = _skill_name(sequence)
        content = _render_skill(sequence, items, date_str)
        skills.append({
            "name": name,
            "sequence": sequence,
            "count": len(items),
            "content": content,
            "file_name": _safe_file_name(name, sequence),
        })

    logger.info(f"[BuildSkills] 最近 {days} 天成功 {sum(len(v) for v in grouped.values())} 条，"
                f"满足 min_count={min_count} 的技能组 {len(skills)} 个")
    if not skills:
        logger.info("[BuildSkills] 暂无可生成的技能（成功轨迹按工具序列聚类后均未达到阈值）")

    if write and skills:
        raw_dir = (experience_config.skill_dir or "./skills").strip()
        out_dir = Path(raw_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for skill in skills:
            path = out_dir / skill["file_name"]
            path.write_text(skill["content"], encoding="utf-8")
            logger.info(f"[BuildSkills] 已写入技能：{path}")
    elif skills:
        for skill in skills:
            logger.info(f"\n===== [dry-run] {skill['file_name']}（{skill['count']} 条） =====\n"
                        f"{skill['content']}")

    return skills


def main():
    parser = argparse.ArgumentParser(description="从成功 ReAct 轨迹生成 SKILL.md")
    parser.add_argument("--days", type=int, default=30, help="统计最近 N 天的轨迹（默认 30）")
    parser.add_argument("--min-count", type=int, default=3, help="同工具序列最少出现次数（默认 3）")
    parser.add_argument("--write", action="store_true", help="写入 skills/ 目录（默认只打印）")
    parser.add_argument("--dry-run", action="store_true", help="仅打印，不写文件（默认行为）")
    args = parser.parse_args()
    build_skills(days=args.days, min_count=args.min_count, write=args.write and not args.dry_run)


if __name__ == "__main__":
    main()

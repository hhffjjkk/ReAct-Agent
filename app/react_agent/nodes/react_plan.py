"""
ReAct 规划节点：让 LLM 输出 Thought / Action（或 Final Answer）。
达到 max_steps 时不再调用 LLM，直接强制生成“转人工”结论（硬上限）。
"""
import json
import re
from typing import Any, Dict, Tuple

from langchain_core.messages import HumanMessage

from app.core.logger import logger
from app.lm.lm_utils import get_llm_client
from app.react_agent.agent_sse import push_agent_event
from app.react_agent.prompt_builder import build_react_prompt, prepare_injections

_TRANSFER_KEYWORDS = ("转人工", "人工工单", "转交人工", "人工处理", "人工排查")


def _history_text(session_id: str) -> str:
    """读取最近对话（复用现有 chat_message），失败不影响主流程"""
    try:
        from app.clients.mongo_history_utils import get_recent_messages
        messages = get_recent_messages(session_id, limit=6)
        if not messages:
            return "无历史对话记录。"
        lines = []
        for m in messages:
            role = "用户" if m.get("role") == "user" else "助手"
            lines.append(f"【{role}】{m.get('text', '')}")
        return "\n".join(lines)
    except Exception:
        return "无历史对话记录。"


def _scratchpad_text(steps) -> str:
    if not steps:
        return "（暂无，尚未调用任何工具）"
    lines = []
    for item in steps:
        obs = item.get("observation") or {}
        obs_text = json.dumps(obs, ensure_ascii=False)
        if len(obs_text) > 1800:
            obs_text = obs_text[:1800] + "...(截断)"
        lines.append(
            f"步骤 {item.get('step')}：\n"
            f"Thought: {item.get('thought', '')}\n"
            f"Action: {item.get('action', '')}\n"
            f"Action Input: {json.dumps(item.get('action_input') or {}, ensure_ascii=False)}\n"
            f"Observation: {obs_text}"
        )
    return "\n\n".join(lines)


def _strip_code_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_react_output(content: str) -> Tuple[str, str, Dict[str, Any], str]:
    """
    解析 LLM 输出，返回 (thought, action, action_input, final_answer)。
    action 为工具名或 "final"；格式非法时 action 返回 "__invalid__"。
    """
    text = _strip_code_fence(content)

    # 兼容模型直接输出 JSON 的情况
    if text.startswith("{"):
        try:
            data = json.loads(text)
            thought = data.get("thought", "") or ""
            final = data.get("final_answer") or data.get("final") or ""
            if final:
                return thought, "final", {}, final
            action = (data.get("action") or "").strip()
            action_input = data.get("action_input") or {}
            if action:
                return thought, action, action_input if isinstance(action_input, dict) else {}, ""
        except Exception:
            pass

    final_match = re.search(r"Final\s*Answer\s*[:：]\s*(.*)", text, re.S | re.I)
    if final_match:
        thought_match = re.search(r"Thought\s*[:：]\s*(.*?)(?=\n\s*Final\s*Answer\s*[:：]|$)", text, re.S | re.I)
        thought = thought_match.group(1).strip() if thought_match else ""
        return thought, "final", {}, final_match.group(1).strip()

    thought_match = re.search(r"Thought\s*[:：]\s*(.*?)(?=\n\s*Action\s*[:：]|$)", text, re.S | re.I)
    thought = thought_match.group(1).strip() if thought_match else ""

    action_match = re.search(r"Action\s*[:：]\s*([A-Za-z_][A-Za-z0-9_]*)", text, re.I)
    if not action_match:
        return thought, "__invalid__", {}, ""

    action = action_match.group(1).strip()
    action_input: Dict[str, Any] = {}
    input_match = re.search(r"Action\s*Input\s*[:：]\s*(\{.*)", text, re.S | re.I)
    if input_match:
        block = input_match.group(1).strip()
        try:
            decoder = json.JSONDecoder()
            parsed, _ = decoder.raw_decode(block)
            if isinstance(parsed, dict):
                action_input = parsed
        except Exception:
            # 兜底：只截取第一对大括号
            start, end = block.find("{"), block.rfind("}")
            if start >= 0 and end > start:
                try:
                    action_input = json.loads(block[start:end + 1])
                except Exception:
                    action_input = {}
    return thought, action, action_input, ""


def _forced_final(state) -> Dict[str, Any]:
    """达到最大步数：强制转人工"""
    answer = (
        f"已执行 {state.get('step', 0)} 步排查，仍未定位到明确根因。"
        "建议转人工工单，由 IT 运维现场检查打印机硬件、驱动与权限配置。"
    )
    logger.warning(f"[AgentReact] max_steps_reached=true session_id={state['session_id']}")
    return {
        "thought": "已达到最大步数硬上限，停止继续调用工具。",
        "action": "final",
        "action_input": {},
        "final_answer": answer,
        "transfer_to_human": True,
        "max_steps_reached": True,
    }


def _injection_payload(state) -> Dict[str, Any]:
    """把注入元数据从 state 复制到节点返回值（LangGraph 状态合并用）"""
    keys = (
        "experience_memory_enabled", "experience_hit", "experience_injected",
        "experience_latency_ms", "experience_ids",
        "skill_injection_enabled", "skill_injected", "skill_name",
        "skill_block", "experience_block",
    )
    return {k: state.get(k) for k in keys if k in state}


def node_react_plan(state):
    """ReAct 规划：输出 thought/action/action_input 或 final_answer"""
    session_id = state["session_id"]
    step = int(state.get("step", 0) or 0)
    max_steps = int(state.get("max_steps", 6) or 6)

    # 硬上限：由条件边再次兜底
    if step >= max_steps:
        result = _forced_final(state)
        push_agent_event(session_id, "thought", {"step": step, "thought": result["thought"]})
        push_agent_event(session_id, "action", {"step": step, "action": "final", "action_input": {}})
        return {**result, **_injection_payload(state)}

    # 仅在进入 ReAct 循环第一步前做一次经验 / 技能检索
    if step == 0:
        meta = prepare_injections(state.get("query", ""))
        state.update(meta)

    prompt = build_react_prompt(
        question=state.get("query", ""),
        history_text=_history_text(session_id),
        scratchpad_text=_scratchpad_text(state.get("scratchpad", [])),
        step=step,
        max_steps=max_steps,
        skill_block=state.get("skill_block", "") or "",
        experience_block=state.get("experience_block", "") or "",
    )
    injection_payload = _injection_payload(state)

    try:
        client = get_llm_client()
        response = client.invoke([HumanMessage(content=prompt)])
        content = response.content or ""
        thought, action, action_input, final_answer = parse_react_output(content)
        logger.info(f"[AgentReact] step={step} thought={thought!r} action={action!r} "
                    f"input={action_input} final={bool(final_answer)}")

        if final_answer:
            transfer = any(k in final_answer for k in _TRANSFER_KEYWORDS)
            push_agent_event(session_id, "thought", {"step": step, "thought": thought})
            push_agent_event(session_id, "action", {"step": step, "action": "final", "action_input": {}})
            return {
                "thought": thought,
                "action": "final",
                "action_input": {},
                "final_answer": final_answer,
                "transfer_to_human": transfer,
                **injection_payload,
            }

        if action == "__invalid__":
            # 格式非法：停止循环，避免空转
            answer = "模型输出格式异常，未能继续自动诊断。建议转人工工单，由 IT 运维协助处理。"
            push_agent_event(session_id, "thought", {"step": step, "thought": thought})
            push_agent_event(session_id, "action", {"step": step, "action": "final", "action_input": {}})
            return {
                "thought": thought,
                "action": "final",
                "action_input": {},
                "final_answer": answer,
                "transfer_to_human": True,
                "error": "invalid_llm_output",
                **injection_payload,
            }

        push_agent_event(session_id, "thought", {"step": step, "thought": thought})
        push_agent_event(session_id, "action", {
            "step": step, "action": action, "action_input": action_input,
        })
        return {
            "thought": thought,
            "action": action,
            "action_input": action_input,
            **injection_payload,
        }

    except Exception as e:
        logger.exception(f"[AgentReact] LLM 规划失败：{e}")
        answer = "诊断模型调用失败，暂时无法继续自动排查。建议转人工工单处理。"
        push_agent_event(session_id, "thought", {"step": step, "thought": "模型调用异常，终止自动诊断。"})
        push_agent_event(session_id, "action", {"step": step, "action": "final", "action_input": {}})
        return {
            "thought": "模型调用异常，终止自动诊断。",
            "action": "final",
            "action_input": {},
            "final_answer": answer,
            "transfer_to_human": True,
            "error": f"llm_failed: {e}",
            **injection_payload,
        }

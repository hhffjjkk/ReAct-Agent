"""
ReAct Prompt 构造与经验/技能注入。

关键约束：两个开关都关闭时，返回的 prompt 必须与改造前完全一致。
实现方式：基础 prompt 仍由 prompts/react_system.md 渲染；仅在存在注入块时，
在“## 可用工具”之前做字符串插入，基础文本本身不做任何改动。
"""
from typing import Any, Dict, Tuple

from app.conf.experience_config import experience_config
from app.core.logger import logger
from app.react_agent.experience_memory import (
    build_experience_block,
    retrieve_experiences,
)
from app.react_agent.prompt_loader import load_md_prompt
from app.react_agent.skill_loader import build_skill_block, select_skill
from app.react_agent.tools import render_tools_for_prompt

_INJECT_MARKER = "## 可用工具"


def _base_prompt(question: str, history_text: str, scratchpad_text: str,
                 step: int, max_steps: int) -> str:
    return load_md_prompt(
        "react_system",
        tools=render_tools_for_prompt(),
        history=history_text,
        scratchpad=scratchpad_text,
        step=step,
        max_steps=max_steps,
        question=question,
    )


def insert_blocks(base_prompt: str, skill_block: str = "",
                  experience_block: str = "") -> str:
    """
    把技能/经验块插入到“## 可用工具”之前。
    顺序：技能块在前（优先级更高），经验块在后。
    两个块都为空时原样返回 base_prompt。
    """
    blocks = [b for b in (skill_block, experience_block) if b]
    if not blocks:
        return base_prompt
    injection = "\n\n".join(blocks) + "\n\n"
    idx = base_prompt.find(_INJECT_MARKER)
    if idx < 0:
        return injection + base_prompt
    return base_prompt[:idx] + injection + base_prompt[idx:]


def build_react_prompt(question: str, history_text: str, scratchpad_text: str,
                       step: int, max_steps: int,
                       skill_block: str = "",
                       experience_block: str = "") -> str:
    """构造最终 ReAct prompt（基础模板 + 可选注入）"""
    base = _base_prompt(question, history_text, scratchpad_text, step, max_steps)
    return insert_blocks(base, skill_block=skill_block, experience_block=experience_block)


def prepare_injections(query: str) -> Dict[str, Any]:
    """
    在 ReAct 第一步前准备注入内容与可观测元数据。
    关闭开关时不触发任何 Milvus / 文件检索。
    """
    meta: Dict[str, Any] = {
        "experience_memory_enabled": experience_config.experience_memory_enabled,
        "experience_hit": False,
        "experience_injected": 0,
        "experience_latency_ms": 0,
        "experience_ids": [],
        "skill_injection_enabled": experience_config.skill_injection_enabled,
        "skill_injected": False,
        "skill_name": "",
        "skill_block": "",
        "experience_block": "",
    }

    if experience_config.skill_injection_enabled:
        try:
            skill = select_skill(query)
            block = build_skill_block(skill)
            if block:
                meta["skill_block"] = block
                meta["skill_injected"] = True
                meta["skill_name"] = skill.get("name", "")
        except Exception as e:
            logger.warning(f"[PromptBuilder] 技能注入失败（已忽略）：{e}")

    if experience_config.experience_memory_enabled:
        try:
            experiences, latency_ms, timed_out = retrieve_experiences(query)
            block = build_experience_block(experiences)
            meta["experience_latency_ms"] = latency_ms
            if block and not timed_out:
                meta["experience_block"] = block
                meta["experience_hit"] = True
                meta["experience_injected"] = len(
                    [x for x in experiences if x.get("success") is True]
                )
                meta["experience_ids"] = [x.get("exp_id") for x in experiences]
        except Exception as e:
            logger.warning(f"[PromptBuilder] 经验注入失败（已忽略）：{e}")

    return meta

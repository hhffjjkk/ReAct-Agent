# -*- coding: utf-8 -*-
"""能力 B：技能加载、在线注入优先级、离线生成测试（不依赖真实 LLM / Milvus）。"""
import time

from app.conf.experience_config import experience_config
from app.react_agent import prompt_builder
from app.react_agent import skill_loader
from app.tool import build_skills as build_skills_module

SKILL_TEXT = """# Skill: 打印队列卡住排查
## 触发条件
错误码 E3 / 队列卡住 / 不出纸
## 工具调用顺序
1. check_device_status — 确认设备在线
2. query_ticket_history — 参考历史工单
## 关键判断
若设备离线且队列积压，先重启设备并清理队列。
## 来源
基于 3 条成功轨迹，最近更新 2026-10-26
"""


def _write_skill(tmp_path, name="print_queue.md", text=SKILL_TEXT):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _experiences():
    return [{
        "exp_id": "exp-1",
        "query_text": "打印机报 E3 错误码",
        "tool_sequence": "check_device_status→query_ticket_history",
        "final_answer": "重启打印机并清理打印队列。",
        "success": True,
        "score": 0.82,
    }]


def test_select_skill_by_keywords(tmp_path, monkeypatch):
    _write_skill(tmp_path)
    monkeypatch.setattr(experience_config, "skill_injection_enabled", True)
    monkeypatch.setattr(experience_config, "skill_dir", str(tmp_path))

    skill = skill_loader.select_skill("打印机报错 E3，任务卡在队列里")
    assert skill is not None
    assert skill["name"] == "打印队列卡住排查"
    assert any(kw == "E3" for kw in skill["keywords"])

    assert skill_loader.select_skill("显示器亮度怎么调") is None


def test_skill_priority_before_experience(tmp_path, monkeypatch):
    """同时命中技能与参考经验时，技能块必须出现在经验块之前"""
    _write_skill(tmp_path)
    monkeypatch.setattr(experience_config, "skill_injection_enabled", True)
    monkeypatch.setattr(experience_config, "skill_dir", str(tmp_path))
    monkeypatch.setattr(experience_config, "experience_memory_enabled", True)
    monkeypatch.setattr(
        prompt_builder, "retrieve_experiences",
        lambda query: (_experiences(), 35, False),
    )

    meta = prompt_builder.prepare_injections("打印机报错 E3，队列卡住")
    assert meta["skill_injected"] is True
    assert meta["experience_hit"] is True

    prompt = prompt_builder.build_react_prompt(
        question="打印机报错 E3，队列卡住",
        history_text="无历史对话记录。",
        scratchpad_text="（暂无，尚未调用任何工具）",
        step=0,
        max_steps=6,
        skill_block=meta["skill_block"],
        experience_block=meta["experience_block"],
    )
    assert prompt.count("【可用技能】") == 1
    assert prompt.index("【可用技能】") < prompt.index("【参考经验】")
    assert prompt.index("【参考经验】") < prompt.index("## 可用工具")


def test_skill_injection_only_one(tmp_path, monkeypatch):
    """命中多个技能时最多注入 1 个"""
    _write_skill(tmp_path, "a.md", SKILL_TEXT)
    _write_skill(tmp_path, "b.md", SKILL_TEXT.replace("打印队列卡住排查", "打印设备离线排查"))
    monkeypatch.setattr(experience_config, "skill_injection_enabled", True)
    monkeypatch.setattr(experience_config, "skill_dir", str(tmp_path))

    skills = skill_loader.load_skills()
    assert len(skills) == 2
    selected = skill_loader.select_skill("打印机 E3 队列卡住")
    assert selected is not None
    block = skill_loader.build_skill_block(selected)
    assert block.count("【可用技能】") == 1


def test_build_skills_offline(tmp_path, monkeypatch):
    """3 条相同工具序列的成功轨迹 -> 生成 1 个 SKILL.md"""
    now = time.time()
    traces = []
    for i in range(3):
        traces.append({
            "session_id": f"skill-sess-{i}",
            "query": "我打印不了，提示错误码 E3",
            "route": "react",
            "max_steps_reached": False,
            "final_answer": "重启打印机并清理打印队列。",
            "created_at": now - i,
            "steps": [
                {"step": 1, "action": "check_device_status"},
                {"step": 2, "action": "query_ticket_history"},
            ],
        })

    monkeypatch.setattr(build_skills_module, "list_agent_traces", lambda limit, since_ts: traces)
    monkeypatch.setattr(build_skills_module, "has_positive_feedback", lambda sid: True)
    monkeypatch.setattr(experience_config, "skill_dir", str(tmp_path))

    skills = build_skills_module.build_skills(days=30, min_count=3, write=True)
    assert len(skills) == 1
    assert skills[0]["count"] == 3
    assert skills[0]["sequence"] == "check_device_status→query_ticket_history"
    content = skills[0]["content"]
    assert content.startswith("# Skill:")
    assert "## 触发条件" in content
    assert "基于 3 条成功轨迹" in content
    assert (tmp_path / skills[0]["file_name"]).exists()

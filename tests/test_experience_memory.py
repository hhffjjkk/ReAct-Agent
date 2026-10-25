# -*- coding: utf-8 -*-
"""能力 A：在线经验记忆测试（全部 mock LLM / Milvus / Mongo）。"""
import asyncio
import time
from types import SimpleNamespace

from app.conf.experience_config import experience_config
from app.react_agent import experience_memory as em
from app.react_agent import prompt_builder
from app.react_agent.prompt_loader import load_md_prompt
from app.react_agent.state import create_agent_default_state
from app.react_agent.tools import render_tools_for_prompt


def _sample_trace(session_id="test-exp-001"):
    return {
        "session_id": session_id,
        "query": "我打印不了，提示错误码 E3",
        "route": "react",
        "max_steps_reached": False,
        "final_answer": "根据历史工单 TK-2026-1001，重启打印机并清理队列。",
        "created_at": time.time(),
        "steps": [
            {"step": 1, "action": "check_device_status", "observation": {"hit": False}},
            {"step": 2, "action": "query_ticket_history", "observation": {"hit": True}},
        ],
    }


def _experiences():
    return [
        {
            "exp_id": "exp-1",
            "query_text": "打印机报 E3 错误码",
            "tool_sequence": "check_device_status→query_ticket_history",
            "final_answer": "重启打印机并清理打印队列，检查纸张与硒鼓。",
            "success": True,
            "score": 0.82,
        },
        {
            "exp_id": "exp-2",
            "query_text": "打印机无法出纸",
            "tool_sequence": "check_device_status",
            "final_answer": "检查队列与设备在线状态。",
            "success": True,
            "score": 0.78,
        },
    ]


def test_disabled_experience_prompt_identical(monkeypatch):
    """开关关闭时，react_plan 构造的 prompt 与改造前逐字符一致（mock LLM 断言）"""
    monkeypatch.setattr(experience_config, "experience_memory_enabled", False)
    monkeypatch.setattr(experience_config, "skill_injection_enabled", False)

    captured = {}

    class FakeLLM:
        def invoke(self, messages):
            captured["prompt"] = messages[0].content
            return SimpleNamespace(content="Thought: 已定位\nFinal Answer: 测试结论")

    import app.react_agent.nodes.react_plan as react_plan
    monkeypatch.setattr(react_plan, "get_llm_client", lambda: FakeLLM())
    monkeypatch.setattr(react_plan, "_history_text", lambda sid: "无历史对话记录。")

    state = create_agent_default_state(
        session_id="test-disabled", query="我打印不了", step=0, max_steps=6,
    )
    react_plan.node_react_plan(state)

    expected = load_md_prompt(
        "react_system",
        tools=render_tools_for_prompt(),
        history="无历史对话记录。",
        scratchpad="（暂无，尚未调用任何工具）",
        step=0,
        max_steps=6,
        question="我打印不了",
    )
    assert captured["prompt"] == expected
    assert "【参考经验】" not in captured["prompt"]
    assert "【可用技能】" not in captured["prompt"]


def test_success_trace_writes_experience(monkeypatch):
    """max_steps_reached=false 且有正确反馈时，写入 success=true"""
    monkeypatch.setattr(experience_config, "experience_memory_enabled", True)
    monkeypatch.setattr(em, "has_positive_feedback", lambda sid: True)
    monkeypatch.setattr(em, "embed_query", lambda q: [0.01] * 1024)

    captured = {}
    import app.clients.experience_milvus_utils as milvus_utils
    monkeypatch.setattr(
        milvus_utils, "insert_experience",
        lambda exp: captured.update(exp) or True,
    )

    ok = asyncio.run(em.write_experience_from_trace(_sample_trace()))
    assert ok is True
    assert captured["success"] is True
    assert captured["tool_sequence"] == "check_device_status→query_ticket_history"
    assert captured["exp_id"].startswith("test-exp-001_")
    assert len(captured["query_embedding"]) == 1024


def test_failed_trace_writes_false(monkeypatch):
    """max_steps_reached=true 时写 success=false（不参与在线检索）"""
    monkeypatch.setattr(experience_config, "experience_memory_enabled", True)
    monkeypatch.setattr(em, "has_positive_feedback", lambda sid: False)
    monkeypatch.setattr(em, "embed_query", lambda q: [0.01] * 1024)

    captured = {}
    import app.clients.experience_milvus_utils as milvus_utils
    monkeypatch.setattr(milvus_utils, "insert_experience", lambda exp: captured.update(exp) or True)

    trace = _sample_trace("test-exp-failed")
    trace["max_steps_reached"] = True
    assert asyncio.run(em.write_experience_from_trace(trace)) is True
    assert captured["success"] is False


def test_experience_injection_contains_top1(monkeypatch):
    """预置 2 条成功经验，断言 prompt 出现【参考经验】且包含 top-1 摘要"""
    monkeypatch.setattr(experience_config, "experience_memory_enabled", True)
    monkeypatch.setattr(experience_config, "experience_memory_topk", 2)
    monkeypatch.setattr(
        prompt_builder, "retrieve_experiences",
        lambda query: (_experiences(), 42, False),
    )

    meta = prompt_builder.prepare_injections("打印机报错 E3 怎么处理")
    assert meta["experience_hit"] is True
    assert meta["experience_injected"] == 2
    assert meta["experience_latency_ms"] == 42

    prompt = prompt_builder.build_react_prompt(
        question="打印机报错 E3 怎么处理",
        history_text="无历史对话记录。",
        scratchpad_text="（暂无，尚未调用任何工具）",
        step=0,
        max_steps=6,
        experience_block=meta["experience_block"],
    )
    assert "【参考经验】" in prompt
    assert "重启打印机并清理打印队列" in prompt
    assert "相似度 0.82" in prompt


def test_failed_experience_not_injected():
    """success=false 的经验不能出现在 prompt 中"""
    exp = dict(_experiences()[0])
    exp["success"] = False
    block = em.build_experience_block([exp])
    assert block == ""

    prompt = prompt_builder.build_react_prompt(
        question="打印机报错 E3", history_text="无历史对话记录。",
        scratchpad_text="（暂无，尚未调用任何工具）", step=0, max_steps=6,
        experience_block=block,
    )
    assert "【参考经验】" not in prompt


def test_timeout_skips_injection(monkeypatch):
    """检索超时（或超过 300ms）时跳过注入，不阻塞主流程"""
    monkeypatch.setattr(experience_config, "experience_memory_enabled", True)
    monkeypatch.setattr(
        prompt_builder, "retrieve_experiences",
        lambda query: ([], 300, True),
    )
    meta = prompt_builder.prepare_injections("打印机报错 E3")
    assert meta["experience_hit"] is False
    assert meta["experience_injected"] == 0
    assert meta["experience_block"] == ""
    assert meta["experience_latency_ms"] == 300


def test_metrics_experience_fields(monkeypatch):
    """/agent/metrics 输出经验记忆命中率、注入次数、平均检索延迟"""
    import app.react_agent.metrics as metrics

    traces = [
        {
            "route": "react", "step_count": 2, "latency_ms": 1000,
            "experience_memory_enabled": True, "experience_hit": True,
            "experience_injected": 2, "experience_latency_ms": 120,
            "skill_injected": True,
        },
        {
            "route": "react", "step_count": 1, "latency_ms": 800,
            "experience_memory_enabled": True, "experience_hit": False,
            "experience_injected": 0, "experience_latency_ms": 80,
            "skill_injected": False,
        },
    ]
    monkeypatch.setattr(metrics, "list_agent_traces", lambda limit, since_ts: traces)
    monkeypatch.setattr(
        "app.clients.mongo_monitor_utils.get_recent_logs",
        lambda limit, since_ts: [],
    )

    result = metrics.build_agent_metrics(days=14)
    summary = result["summary"]
    assert summary["experience_memory_hit_rate"] == 0.5
    assert summary["experience_memory_injected_count"] == 2
    assert summary["avg_retrieval_latency_ms"] == 100
    assert summary["skill_injected_count"] == 1
    assert result["experience_memory"]["hit_count"] == 1


def test_single_rag_branch_unchanged(monkeypatch):
    """/agent/query 走 single_rag 时行为不变（不触发经验/技能注入）"""
    import app.query_process.agent.main_graph as query_graph
    import app.react_agent.nodes.generate_answer as gen
    import app.react_agent.nodes.react_plan as react_plan
    from app.react_agent.graph import agent_app

    class FakeQueryApp:
        def invoke(self, state):
            return {"answer": "RAG_ANSWER"}

    monkeypatch.setattr(query_graph, "query_app", FakeQueryApp())
    monkeypatch.setattr(gen, "save_agent_trace", lambda trace: True)
    monkeypatch.setattr(gen, "schedule_experience_write", lambda trace: None)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("single_rag 不应构造 ReAct Prompt")

    monkeypatch.setattr(react_plan, "build_react_prompt", _fail_if_called)

    state = create_agent_default_state(
        session_id="test-single-rag", query="打印机驱动怎么安装",
        max_steps=6, start_ts=time.time(),
    )
    result = agent_app.invoke(state)
    assert result["route"] == "single_rag"
    assert result["final_answer"] == "RAG_ANSWER"

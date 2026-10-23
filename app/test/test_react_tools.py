"""
ReAct 工具与图结构的白盒测试（不依赖 LLM）。
运行：uv run python -m unittest app.test.test_react_tools -v
"""
import time
import unittest
import uuid


def _fake_tool_output():
    """构造一个始终调用工具的 LLM 输出，用于验证最大步数硬上限"""
    from types import SimpleNamespace

    class FakeClient:
        def invoke(self, messages):
            return SimpleNamespace(content=(
                "Thought: 继续排查设备状态\n"
                "Action: check_device_status\n"
                'Action Input: {"device_id": "PRN-001"}'
            ))

    return lambda *args, **kwargs: FakeClient()

from app.react_agent.nodes.intent_router import _rule_route
from app.react_agent.nodes.react_plan import parse_react_output
from app.react_agent.tools import (
    TOOL_DEFINITIONS,
    execute_tool,
    query_ticket_history,
    search_knowledge_base,
    check_device_status,
    check_user_permission,
)


class TestMockTools(unittest.TestCase):
    def test_device_status_by_id(self):
        result = check_device_status(device_id="PRN-002")
        self.assertTrue(result["hit"])
        self.assertFalse(result["online"])
        self.assertEqual(result["error_code"], "E3")
        self.assertEqual(result["queue_length"], 3)

    def test_device_status_by_name(self):
        result = check_device_status(printer_name="5F-研发-佳能-iR2625")
        self.assertTrue(result["hit"])
        self.assertTrue(result["online"])

    def test_device_status_not_found(self):
        result = check_device_status(device_id="PRN-999")
        self.assertFalse(result["hit"])
        self.assertIn("未找到", result["error"])

    def test_user_permission_ok_and_driver_mismatch(self):
        result = check_user_permission(user_id="U1001", device_id="PRN-001")
        self.assertTrue(result["hit"])
        self.assertTrue(result["has_permission"])
        self.assertNotEqual(result["driver_version"], result["expected_version"])

    def test_user_permission_denied(self):
        result = check_user_permission(user_id="U1002", device_id="PRN-002")
        self.assertTrue(result["hit"])
        self.assertFalse(result["has_permission"])

    def test_user_permission_not_found(self):
        result = check_user_permission(user_id="U9999", device_id="PRN-001")
        self.assertFalse(result["hit"])
        self.assertIn("未找到", result["error"])


class TestKnowledgeBaseTool(unittest.TestCase):
    def test_search_structure_and_filter_inheritance(self):
        result = search_knowledge_base("打印机驱动怎么安装")
        self.assertIsInstance(result, dict)
        self.assertIn("hit", result)
        self.assertIsInstance(result["hit"], bool)
        self.assertIsInstance(result.get("results", []), list)
        for item in result.get("results", []):
            for key in ("content", "source_doc", "publish_date", "expire_date"):
                self.assertIn(key, item)

        # 白盒校验：过滤表达式继承版本 + 时效过滤
        from app.clients.milvus_utils import build_chunk_filter_expr
        expr = build_chunk_filter_expr([], now_ts=1)
        self.assertIn("is_latest == true", expr)
        self.assertIn("expire_date", expr)


class TestTicketTool(unittest.TestCase):
    def test_ticket_history_hit(self):
        result = query_ticket_history("打印不了，提示错误码 E3，任务卡住")
        self.assertTrue(result["hit"])
        self.assertGreater(len(result["tickets"]), 0)
        for key in ("ticket_id", "symptom", "resolution", "resolved_at"):
            self.assertIn(key, result["tickets"][0])

    def test_ticket_history_no_hit(self):
        result = query_ticket_history("今天天气怎么样")
        self.assertFalse(result["hit"])


class TestDispatch(unittest.TestCase):
    def test_unknown_tool(self):
        result = execute_tool("not_exist_tool", {})
        self.assertFalse(result["hit"])
        self.assertIn("工具不存在", result["error"])

    def test_invalid_action_input(self):
        result = execute_tool("search_knowledge_base", "not-a-dict")
        self.assertFalse(result["hit"])
        self.assertIn("JSON 对象", result["error"])

    def test_missing_required_param(self):
        result = execute_tool("check_user_permission", {"user_id": "U1001"})
        self.assertFalse(result["hit"])
        self.assertIn("缺少参数", result["error"])

    def test_tool_definitions_fixed(self):
        names = sorted(t["name"] for t in TOOL_DEFINITIONS)
        self.assertEqual(names, sorted([
            "search_knowledge_base", "query_ticket_history",
            "check_device_status", "check_user_permission",
        ]))
        # description 必须是“何时使用”
        for tool in TOOL_DEFINITIONS:
            self.assertTrue(tool["description"].startswith("当"))


class TestParserAndRouting(unittest.TestCase):
    def test_parse_tool_action(self):
        text = (
            "Thought: 需要确认设备状态\n"
            "Action: check_device_status\n"
            'Action Input: {"device_id": "PRN-002"}'
        )
        thought, action, args, final = parse_react_output(text)
        self.assertIn("设备状态", thought)
        self.assertEqual(action, "check_device_status")
        self.assertEqual(args["device_id"], "PRN-002")
        self.assertEqual(final, "")

    def test_parse_final_answer(self):
        text = "Thought: 已定位\nFinal Answer: 建议重启打印机并清理队列。"
        _, action, _, final = parse_react_output(text)
        self.assertEqual(action, "final")
        self.assertIn("重启打印机", final)

    def test_parse_json_output(self):
        text = '{"thought": "查询权限", "action": "check_user_permission", "action_input": {"user_id": "U1001", "device_id": "PRN-001"}}'
        _, action, args, _ = parse_react_output(text)
        self.assertEqual(action, "check_user_permission")
        self.assertEqual(args["user_id"], "U1001")

    def test_rule_route(self):
        self.assertEqual(_rule_route("打印机驱动怎么安装"), "single_rag")
        self.assertEqual(_rule_route("我打印不了，提示错误码 E3"), "react")


class TestGraphStructure(unittest.TestCase):
    def test_graph_nodes(self):
        from app.react_agent.graph import agent_app
        nodes = set(agent_app.get_graph().nodes.keys())
        for name in ("intent_router", "single_rag", "react_plan",
                     "execute_tool", "observe", "generate_answer"):
            self.assertIn(name, nodes)


class TestMaxSteps(unittest.TestCase):
    def test_max_steps_hard_cap_and_trace(self):
        """LLM 永远只调用工具时，必须在 6 步后强制转人工并写入 agent_traces"""
        import app.react_agent.nodes.react_plan as react_plan_module
        from app.react_agent.graph import agent_app
        from app.react_agent.state import create_agent_default_state
        from app.react_agent.trace_store import get_agent_trace, _collection

        session_id = f"test-maxsteps-{uuid.uuid4()}"
        original = react_plan_module.get_llm_client
        react_plan_module.get_llm_client = _fake_tool_output()
        try:
            state = create_agent_default_state(
                session_id=session_id,
                query="我打印不了，一直卡住",
                max_steps=6,
                start_ts=time.time(),
            )
            result = agent_app.invoke(state)
            self.assertTrue(result["max_steps_reached"])
            self.assertTrue(result["transfer_to_human"])
            self.assertEqual(result["step"], 6)
            self.assertIn("人工", result["final_answer"])

            trace = get_agent_trace(session_id)
            self.assertIsNotNone(trace)
            self.assertTrue(trace["max_steps_reached"])
            self.assertEqual(trace["step_count"], 6)
            self.assertEqual(trace["route"], "react")
            self.assertEqual(len(trace["steps"]), 6)
        finally:
            react_plan_module.get_llm_client = original
            _collection().delete_one({"session_id": session_id})


class TestAgentSwitch(unittest.TestCase):
    def test_agent_disabled_returns_404(self):
        """AGENT_ENABLED=false 时新增接口统一 404"""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.react_agent import router as router_module

        app = FastAPI()
        app.include_router(router_module.agent_router)
        client = TestClient(app)

        original = router_module.agent_config.enabled
        router_module.agent_config.enabled = False
        try:
            self.assertEqual(client.get("/agent/metrics").status_code, 404)
            self.assertEqual(client.get("/agent/monitor.html").status_code, 404)
            self.assertEqual(client.post("/agent/query", json={"query": "打印不了"}).status_code, 404)
            self.assertEqual(client.get("/agent/history/any").status_code, 404)
        finally:
            router_module.agent_config.enabled = original


if __name__ == "__main__":
    unittest.main(verbosity=2)

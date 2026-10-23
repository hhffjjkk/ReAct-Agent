"""
ReAct 诊断 Agent 的 LangGraph 图。

拓扑（对应任务文档 2.3）：
  entry
  → intent_router
  ├─ single_rag   （事实型，复用原单轮 RAG 图）
  └─ react_plan
     ├─ execute_tool → observe → react_plan   （conditional edge 表达循环）
     └─ generate_answer → END

终止条件双重实现：
  ① LLM 输出 Final Answer → generate_answer
  ② step >= max_steps（默认 6）→ react_plan 强制转人工 → generate_answer
循环全部由 LangGraph 条件边驱动，节点内部没有 while 循环。
"""
from langgraph.graph import END, StateGraph

from app.react_agent.nodes.execute_tool import node_execute_tool
from app.react_agent.nodes.generate_answer import node_generate_answer
from app.react_agent.nodes.intent_router import node_intent_router
from app.react_agent.nodes.observe import node_observe
from app.react_agent.nodes.react_plan import node_react_plan
from app.react_agent.nodes.single_rag import node_single_rag
from app.react_agent.state import AgentGraphState

builder = StateGraph(AgentGraphState)

builder.add_node("intent_router", node_intent_router)
builder.add_node("single_rag", node_single_rag)
builder.add_node("react_plan", node_react_plan)
builder.add_node("execute_tool", node_execute_tool)
builder.add_node("observe", node_observe)
builder.add_node("generate_answer", node_generate_answer)

builder.set_entry_point("intent_router")


def route_after_intent(state: AgentGraphState) -> str:
    return "react_plan" if state.get("route") == "react" else "single_rag"


builder.add_conditional_edges("intent_router", route_after_intent, {
    "single_rag": "single_rag",
    "react_plan": "react_plan",
})

builder.add_edge("single_rag", "generate_answer")


def route_after_plan(state: AgentGraphState) -> str:
    """条件边 ①：Final Answer 终止；②：最大步数硬上限终止"""
    if state.get("final_answer") or state.get("action") in ("final", ""):
        return "generate_answer"
    if int(state.get("step", 0) or 0) >= int(state.get("max_steps", 6) or 6):
        return "generate_answer"
    return "execute_tool"


builder.add_conditional_edges("react_plan", route_after_plan, {
    "execute_tool": "execute_tool",
    "generate_answer": "generate_answer",
})

builder.add_edge("execute_tool", "observe")
builder.add_edge("observe", "react_plan")
builder.add_edge("generate_answer", END)

agent_app = builder.compile()

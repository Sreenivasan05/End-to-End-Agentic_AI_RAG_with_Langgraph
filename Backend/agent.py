from models import AgentState
from typing import Literal
from langgraph.graph import StateGraph, END
from chains import rag_node, web_node, router_node, answer_node
from langgraph.checkpoint.memory import MemorySaver

def from_router(st:AgentState) -> Literal["rag","web","answer","end"]:
    return st["route"]

def after_rag(st:AgentState) -> Literal["answer","web"]:
    return st["route"]

def after_web() -> Literal["answer"]:
    return "answer"

def build_agent():
    """
    Builds and compiles the LangGraph agent.
    """
    g = StateGraph(AgentState)

    g.add_node("router",router_node)
    g.add_node("rag_lookup",rag_node)
    g.add_node("web_search",web_node)
    g.add_node("answer",answer_node)

    g.set_entry_point("router")

    g.add_conditional_edges(
        "router",
        from_router,
        {
            "rag" : "rag_lookup",
            "web" : "web_search",
            "answer" : "answer",
            "end" : END
        }
    )

    g.add_conditional_edges(
        "rag_lookup",
        after_rag,
        {
            "answer" : "answer",
            "web" : "web_search"
        }
    )

    g.add_edge("web_search","answer")
    g.add_edge("answer",END)

    agent = g.compile(checkpointer=MemorySaver())

    return agent

rag_agent = build_agent()

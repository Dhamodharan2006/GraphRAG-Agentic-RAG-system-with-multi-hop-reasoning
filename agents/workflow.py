from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.nodes import (
    node_query_planner,
    node_context_retriever,
    node_graph_explorer,
    node_synthesizer
)


def route_by_strategy(state: AgentState) -> str:
    if state.strategy == "vector_only":
        return "retriever"
    elif state.strategy == "graph_only":
        return "explorer"
    return "retriever"  # hybrid starts with retriever


def after_retriever(state: AgentState) -> str:
    if state.strategy == "hybrid" and "graph_explore" in state.plan:
        return "explorer"
    return "synthesizer"


def after_explorer(state: AgentState) -> str:
    return "synthesizer"


def build_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", node_query_planner)
    workflow.add_node("retriever", node_context_retriever)
    workflow.add_node("explorer", node_graph_explorer)
    workflow.add_node("synthesizer", node_synthesizer)

    workflow.set_entry_point("planner")

    workflow.add_conditional_edges("planner", route_by_strategy, {
        "retriever": "retriever",
        "explorer": "explorer"
    })
    workflow.add_conditional_edges("retriever", after_retriever, {
        "explorer": "explorer",
        "synthesizer": "synthesizer"
    })
    workflow.add_conditional_edges("explorer", after_explorer, {
        "synthesizer": "synthesizer"
    })
    workflow.add_edge("synthesizer", END)

    return workflow.compile()


agent_app = build_agent()
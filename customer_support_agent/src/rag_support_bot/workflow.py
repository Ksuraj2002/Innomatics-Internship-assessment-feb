from __future__ import annotations

from typing import Callable

from langgraph.graph import END, START, StateGraph

from .config import get_settings
from .hitl import escalate_to_human
from .intent import detect_intent
from .llm import AnswerGenerator
from .retrieval import SupportRetriever
from .routing import RoutingInput, choose_route
from .state import SupportState


def _confidence_score(query: str, chunks: list[str]) -> float:
    if not chunks:
        return 0.0

    query_terms = {word for word in query.lower().split() if len(word) > 2}
    chunk_text = " ".join(chunks).lower()
    overlap = sum(1 for term in query_terms if term in chunk_text)
    lexical_score = overlap / max(len(query_terms), 1)

    coverage_score = min(len(chunks) / 4.0, 1.0)
    return round((0.7 * lexical_score) + (0.3 * coverage_score), 3)


def build_support_graph() -> Callable[[SupportState], SupportState]:
    settings = get_settings()
    retriever = SupportRetriever()
    generator = AnswerGenerator()

    def processing_node(state: SupportState) -> SupportState:
        query = state["query"]
        intent = detect_intent(query)
        docs = retriever.retrieve(query)
        chunks = [doc.page_content for doc in docs]
        confidence = _confidence_score(query, chunks)
        draft = generator.generate(query, chunks)

        decision = choose_route(
            RoutingInput(intent=intent, confidence=confidence, chunk_count=len(chunks)),
            confidence_threshold=settings.confidence_threshold,
        )

        return {
            **state,
            "intent": intent,
            "retrieved_docs": docs,
            "retrieved_chunks": chunks,
            "confidence": confidence,
            "draft_answer": draft,
            "route": decision.route,
            "escalation_reason": decision.reason,
        }

    def output_node(state: SupportState) -> SupportState:
        return {**state, "final_answer": state.get("draft_answer", "")}

    def hitl_node(state: SupportState) -> SupportState:
        human_reply = escalate_to_human(
            query=state["query"],
            reason=state.get("escalation_reason", "unspecified"),
            draft_answer=state.get("draft_answer", ""),
        )
        if not human_reply:
            human_reply = (
                "A human agent has been notified and will follow up shortly. "
                "Please share your contact details for callback."
            )
        return {**state, "human_response": human_reply, "final_answer": human_reply}

    def route_after_processing(state: SupportState) -> str:
        return state.get("route", "auto_reply")

    graph = StateGraph(SupportState)
    graph.add_node("processing", processing_node)
    graph.add_node("output", output_node)
    graph.add_node("hitl", hitl_node)

    graph.add_edge(START, "processing")
    graph.add_conditional_edges(
        "processing",
        route_after_processing,
        {"auto_reply": "output", "escalate_human": "hitl"},
    )
    graph.add_edge("output", END)
    graph.add_edge("hitl", END)

    compiled = graph.compile()
    return compiled.invoke

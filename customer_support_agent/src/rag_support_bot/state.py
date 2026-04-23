from __future__ import annotations

from typing import Any, Literal, TypedDict


Route = Literal["auto_reply", "escalate_human"]


class SupportState(TypedDict, total=False):
    query: str
    intent: str
    retrieved_docs: list[Any]
    retrieved_chunks: list[str]
    confidence: float
    route: Route
    escalation_reason: str
    draft_answer: str
    final_answer: str
    human_response: str

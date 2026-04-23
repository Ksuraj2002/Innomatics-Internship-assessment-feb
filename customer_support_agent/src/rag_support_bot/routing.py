from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingDecision:
    route: str
    reason: str


@dataclass(frozen=True)
class RoutingInput:
    intent: str
    confidence: float
    chunk_count: int


def choose_route(data: RoutingInput, confidence_threshold: float) -> RoutingDecision:
    if data.intent == "escalation":
        return RoutingDecision(route="escalate_human", reason="user_or_intent_requested_human")

    if data.chunk_count == 0:
        return RoutingDecision(route="escalate_human", reason="missing_context")

    if data.confidence < confidence_threshold:
        return RoutingDecision(route="escalate_human", reason="low_confidence")

    return RoutingDecision(route="auto_reply", reason="sufficient_context")

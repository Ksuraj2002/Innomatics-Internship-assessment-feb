from src.rag_support_bot.routing import RoutingInput, choose_route


def test_escalates_when_intent_is_escalation() -> None:
    decision = choose_route(
        RoutingInput(intent="escalation", confidence=0.9, chunk_count=4),
        confidence_threshold=0.45,
    )
    assert decision.route == "escalate_human"


def test_escalates_on_missing_context() -> None:
    decision = choose_route(
        RoutingInput(intent="general", confidence=0.8, chunk_count=0),
        confidence_threshold=0.45,
    )
    assert decision.route == "escalate_human"


def test_escalates_on_low_confidence() -> None:
    decision = choose_route(
        RoutingInput(intent="general", confidence=0.2, chunk_count=4),
        confidence_threshold=0.45,
    )
    assert decision.route == "escalate_human"


def test_auto_reply_when_confident_and_context_available() -> None:
    decision = choose_route(
        RoutingInput(intent="general", confidence=0.7, chunk_count=3),
        confidence_threshold=0.45,
    )
    assert decision.route == "auto_reply"

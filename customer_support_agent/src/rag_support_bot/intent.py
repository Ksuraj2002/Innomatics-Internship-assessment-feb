from __future__ import annotations


def detect_intent(query: str) -> str:
    text = query.lower()

    escalation_words = [
        "angry",
        "complaint",
        "lawsuit",
        "legal",
        "manager",
        "human",
        "agent",
        "refund not received",
    ]
    billing_words = ["bill", "invoice", "payment", "charged", "refund", "pricing"]
    account_words = ["login", "password", "account", "signin", "sign in", "reset"]
    technical_words = ["error", "bug", "crash", "issue", "not working", "failed"]

    if any(word in text for word in escalation_words):
        return "escalation"
    if any(word in text for word in billing_words):
        return "billing"
    if any(word in text for word in account_words):
        return "account"
    if any(word in text for word in technical_words):
        return "technical"
    return "general"

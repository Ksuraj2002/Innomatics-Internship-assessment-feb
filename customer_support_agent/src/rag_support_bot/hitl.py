from __future__ import annotations


def escalate_to_human(query: str, reason: str, draft_answer: str) -> str:
    print("\n--- HUMAN ESCALATION REQUIRED ---")
    print(f"Reason: {reason}")
    print(f"User query: {query}")
    if draft_answer:
        print(f"AI draft: {draft_answer}")
    print("Enter human agent response:")
    return input("> ").strip()

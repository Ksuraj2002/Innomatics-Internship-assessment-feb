from __future__ import annotations

from .workflow import build_support_graph


def run_cli() -> None:
    invoke = build_support_graph()

    print("Customer Support Assistant (RAG + LangGraph + HITL)")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("User> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Session ended.")
            break

        result = invoke({"query": query})

        print("\nAssistant>")
        print(result.get("final_answer", "No answer generated."))
        print(f"\n[Intent={result.get('intent')} | Confidence={result.get('confidence')} | Route={result.get('route')}]\n")


if __name__ == "__main__":
    run_cli()

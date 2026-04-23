from __future__ import annotations

from typing import Sequence

from langchain_core.prompts import ChatPromptTemplate

from .config import get_settings


class AnswerGenerator:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._llm = None

        if self.settings.gemini_api_key:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._llm = ChatGoogleGenerativeAI(
                model=self.settings.llm_model,
                google_api_key=self.settings.gemini_api_key,
                temperature=0,
            )

        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a customer support assistant. Answer strictly from the context. "
                    "If the context is insufficient, say so briefly.",
                ),
                (
                    "human",
                    "User query: {query}\n\nRetrieved context:\n{context}\n\n"
                    "Return a concise, actionable response.",
                ),
            ]
        )

    def generate(self, query: str, chunks: Sequence[str]) -> str:
        context = "\n\n".join(chunks)

        if self._llm is None:
            if not context.strip():
                return "I do not have enough verified context from the knowledge base to answer this safely."
            return (
                "Based on the knowledge base, here is the best available guidance:\n"
                f"{context[:1200]}\n\n"
                "If this does not fully resolve your issue, I can escalate to a human agent."
            )

        chain = self._prompt | self._llm
        response = chain.invoke({"query": query, "context": context})
        return response.content

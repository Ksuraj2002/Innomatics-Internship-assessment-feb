from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    persist_directory: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    retriever_k: int = int(os.getenv("RETRIEVER_K", "4"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
    llm_model: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    gemini_api_key: str | None = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


def get_settings() -> Settings:
    return Settings()

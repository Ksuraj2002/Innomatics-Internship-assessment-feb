from __future__ import annotations

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from .config import get_settings


class SupportRetriever:
    def __init__(self) -> None:
        settings = get_settings()
        embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)
        self._store = Chroma(
            persist_directory=settings.persist_directory,
            embedding_function=embeddings,
            collection_name="customer_support_kb",
        )
        self._retriever = self._store.as_retriever(search_kwargs={"k": settings.retriever_k})

    def retrieve(self, query: str):
        return self._retriever.invoke(query)

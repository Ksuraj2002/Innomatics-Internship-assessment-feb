from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

from .config import get_settings


def ingest_pdf_to_chroma(pdf_path: str) -> int:
    settings = get_settings()

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.persist_directory,
        collection_name="customer_support_kb",
    )
    return len(chunks)

from __future__ import annotations

import argparse

from src.rag_support_bot.ingestion import ingest_pdf_to_chroma


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a PDF knowledge base into ChromaDB.")
    parser.add_argument("--pdf", required=True, help="Path to the source PDF file.")
    args = parser.parse_args()

    chunk_count = ingest_pdf_to_chroma(args.pdf)
    print(f"Ingestion complete. Stored {chunk_count} chunks.")


if __name__ == "__main__":
    main()

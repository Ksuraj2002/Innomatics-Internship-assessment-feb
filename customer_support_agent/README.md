# RAG-Based Customer Support Assistant (LangGraph + HITL)

This project implements a Retrieval-Augmented Generation (RAG) customer support assistant that:

- Ingests a PDF knowledge base
- Chunks content and stores embeddings in ChromaDB
- Retrieves relevant context for user queries
- Uses LangGraph for workflow orchestration
- Applies conditional routing based on intent and confidence
- Escalates to Human-in-the-Loop (HITL) when required

## Project Structure

```text
.
|-- data/
|-- docs/
|   |-- HLD.md
|   |-- LLD.md
|   `-- Technical_Documentation.md
|-- src/
|   `-- rag_support_bot/
|       |-- config.py
|       |-- ingestion.py
|       |-- intent.py
|       |-- llm.py
|       |-- hitl.py
|       |-- retrieval.py
|       |-- routing.py
|       |-- state.py
|       `-- workflow.py
|-- tests/
|   `-- test_routing.py
|-- ingest.py
|-- run.py
|-- requirements.txt
`-- .env.example
```

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and update values as needed.
4. Place your knowledge base PDF under `data/`, for example `data/customer_support_kb.pdf`.

## Run

### Step 1: Ingest PDF

```bash
python ingest.py --pdf data/customer_support_kb.pdf
```

### Step 2: Start assistant

```bash
python run.py
```

## Query Flow

1. User enters question.
2. Processing node detects intent.
3. Retriever fetches relevant chunks from ChromaDB.
4. System computes confidence and creates draft answer.
5. Conditional routing:
   - `auto_reply` -> Output node returns AI answer.
   - `escalate_human` -> HITL node collects agent response.

## Mandatory Concept Mapping

- What is RAG: See `docs/Technical_Documentation.md`.
- PDF -> Chunk -> ChromaDB: Implemented in `src/rag_support_bot/ingestion.py`.
- Query retrieval: `src/rag_support_bot/retrieval.py`.
- Graph workflow: `src/rag_support_bot/workflow.py`.
- 2-node core flow: `processing -> output` on standard path.
- Conditional routing: `src/rag_support_bot/routing.py` + graph conditional edges.
- Customer support use case: CLI workflow in `src/rag_support_bot/main.py`.
- HITL escalation: `src/rag_support_bot/hitl.py`.

## Testing

```bash
pytest -q
```

## Exporting Deliverables to PDF

If Pandoc is installed:

```bash
pandoc docs/HLD.md -o docs/HLD.pdf
pandoc docs/LLD.md -o docs/LLD.pdf
pandoc docs/Technical_Documentation.md -o docs/Technical_Documentation.pdf
```

If Pandoc is not available, open each Markdown file in VS Code and use Print to PDF.

Alternative (automated PDF generation without Pandoc):

```bash
python scripts/export_docs_pdf.py
```

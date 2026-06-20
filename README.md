# Agentic RAG for Healthcare Accreditation

A layered retrieval-augmented generation project for searching DNV NIAHO
hospital accreditation requirements.

## Current Pipeline

```text
PDF
  -> page extraction
  -> chapter and section detection
  -> overlapping word chunks
  -> BGE-M3 embeddings
  -> LLM query router
  -> semantic search + BM25 keyword search
  -> reciprocal-rank fusion
  -> BGE reranking
  -> grounded Gemini answer with citations
  -> RAGService structured response
```

The chunking strategy preserves chapter, section, and page metadata. By
default, chunks contain up to 900 words with 150 words of overlap.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Place the source PDF in `data/`. PDFs, generated embeddings, and environment
files are intentionally excluded from Git.

## Preview the Data Layer

```bash
python scripts/preview_data_layer.py --show 5
```

## Preview the Embedding Build

The embedding script selects the first three chapters by default.

```bash
python scripts/build_vector_index.py --dry-run
```

For a smaller test:

```bash
python scripts/build_vector_index.py \
  --max-chapters 1 \
  --chunk-words 300 \
  --overlap-words 50 \
  --batch-size 1
```

The generated index is stored locally in:

```text
.rag_index/embeddings.npz
.rag_index/metadata.json
```

## Orchestration Service

All query-time layers are coordinated through one reusable entry point:

```python
from rag_chatbot.rag_service import RAGService

service = RAGService.from_index_dir(".rag_index")
response = service.ask("What is quality management?")

print(response.search_mode)
print(response.answer)
print(response.results)
```

The CLI and future API layer both use this same service.

## FastAPI

Start the local API:

```bash
uvicorn rag_chatbot.api:app --host 127.0.0.1 --port 8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Ask a question:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the quality management responsibilities?",
    "search_mode": "auto",
    "candidate_k": 10,
    "top_k": 3,
    "rerank": true,
    "generate_answer": true
  }'
```

## Search the Vector Index

```bash
python scripts/search_vector_index.py \
  "What are the quality management responsibilities?"
```

By default, Gemini first chooses `semantic`, `keyword`, or `hybrid` retrieval
from the query. If routing is unavailable, the pipeline safely falls back to
hybrid search. The selected candidates are then reranked with
`BAAI/bge-reranker-v2-m3`. The top 3 chunks are sent to Gemini for a grounded
answer with section and page citations.

Set `GOOGLE_API_KEY` in `.env` or your shell to enable automatic routing.
Manual modes remain available for debugging:

To inspect retrieval without generating an answer:

```bash
python scripts/search_vector_index.py \
  "What are the quality management responsibilities?" \
  --show-results-only
```

Compare the three retrieval modes without reranking:

```bash
python scripts/search_vector_index.py \
  "What are the quality management responsibilities?" \
  --search-mode hybrid \
  --no-rerank

python scripts/search_vector_index.py \
  "QM.1 SR.1a QAPI" \
  --search-mode keyword \
  --no-rerank

python scripts/search_vector_index.py \
  "What are the quality management responsibilities?" \
  --search-mode semantic \
  --no-rerank
```

Tune the two retrieval stages independently:

```bash
python scripts/search_vector_index.py \
  "What are the quality management responsibilities?" \
  --candidate-k 15 \
  --top-k 5 \
  --semantic-weight 1.0 \
  --keyword-weight 1.0
```

## Retrieval Models

The project uses the open-weight
[`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) model through
`sentence-transformers` for vector retrieval. It uses
[`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3)
to score each query and candidate passage together before returning results.

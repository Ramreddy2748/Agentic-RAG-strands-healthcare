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

Retrieval depth is selected automatically from the routed mode:

```text
keyword  -> 5 candidates
semantic -> 8 candidates
hybrid   -> 12 candidates
```

The answer is returned in a clinician-friendly structure:

```json
{
  "summary": {"text": "Direct answer", "citations": [1]},
  "key_requirements": [
    {"text": "Requirement", "citations": [1, 2]}
  ],
  "clinical_actions": [
    {"text": "Practical action", "citations": [2]}
  ],
  "limitations": null
}
```

Citation numbers map directly to the authoritative metadata in `sources`.

## Observability

Every query includes a request ID, candidate counts, and stage timings:

```json
{
  "request_id": "d62d...",
  "timings": {
    "routing_ms": 420.1,
    "retrieval_ms": 2100.4,
    "fusion_ms": 0.08,
    "reranking_ms": 8400.2,
    "answer_generation_ms": 1800.5,
    "total_ms": 12721.28
  }
}
```

The API also returns the request ID in the `X-Request-ID` header. Server logs
are emitted as JSON and include the selected mode, candidate counts, returned
sections, timings, and failures. Configure verbosity with `LOG_LEVEL`.

## Model Lifecycle

FastAPI loads BGE-M3 and the BGE reranker once during application startup,
warms both models, and reuses them for every query. This removes repeated model
loading from request latency.

```text
PRELOAD_MODELS=true
MAX_CONCURRENT_REQUESTS=1
```

The default concurrency of one protects CPU and memory while shared Torch
models are running. Increase it only after testing on the target machine.

## Docker

The image contains the API and Python dependencies. The generated vector index,
Gemini credentials, and Hugging Face model cache remain outside the image.

Build the image:

```bash
docker build -t agentic-healthcare-rag:local .
```

Run it directly:

```bash
docker run --rm \
  --name agentic-healthcare-rag \
  --env-file .env \
  -p 8000:8000 \
  -v "$PWD/.rag_index:/app/.rag_index:ro" \
  -v rag-huggingface-cache:/home/appuser/.cache/huggingface \
  agentic-healthcare-rag:local
```

Or use Docker Compose:

```bash
docker compose up --build
```

The first container startup downloads and warms the BGE models. Later starts
reuse the `huggingface-cache` volume. Verify the container at:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

Stop Compose:

```bash
docker compose down
```

Do not copy `.env`, PDFs, `.rag_index`, or model caches into the image. In AWS,
the same container can receive secrets from Secrets Manager and the index from
S3 or a mounted volume.

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

Hybrid fusion weights remain configurable with `--semantic-weight` and
`--keyword-weight`.

## Retrieval Models

The project uses the open-weight
[`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) model through
`sentence-transformers` for vector retrieval. It uses
[`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3)
to score each query and candidate passage together before returning results.

# Agentic RAG for Healthcare Accreditation

[![CI](https://github.com/Ramreddy2748/Agentic-RAG-strands-healthcare/actions/workflows/ci.yml/badge.svg)](https://github.com/Ramreddy2748/Agentic-RAG-strands-healthcare/actions/workflows/ci.yml)
[![Security](https://github.com/Ramreddy2748/Agentic-RAG-strands-healthcare/actions/workflows/security.yml/badge.svg)](https://github.com/Ramreddy2748/Agentic-RAG-strands-healthcare/actions/workflows/security.yml)

An end-to-end healthcare RAG workspace for grounded clinical and accreditation
questions. The project indexes PDFs, CSVs, and JSON files, stores embeddings in
MongoDB Atlas Vector Search or a local vector index, retrieves evidence with
semantic, keyword, or hybrid search, and generates cited answers through Gemini.

## What Is Built

```text
Source documents
  -> extraction for PDF, CSV, and JSON
  -> page/row/value normalization
  -> chapter/section-aware chunking with overlap
  -> BGE-M3 embeddings
  -> local .rag_index or MongoDB Vector Search

User query
  -> security checks
  -> LLM router chooses semantic, keyword, or hybrid search
  -> semantic retrieval + BM25 keyword retrieval
  -> reciprocal-rank fusion for hybrid search
  -> optional BGE reranking
  -> Gemini cited clinical answer
  -> optional Strands verification
  -> FastAPI response
  -> Next.js frontend
```

The default chunking strategy uses up to `900` words per chunk with `150` words
of overlap. Chunk metadata preserves source ID, source path, page range, chapter
title, section title, and chunk ID.

## Repository Layout

```text
rag_chatbot/       Python RAG layers and FastAPI app
frontend/          Next.js web app with login, upload, and chat UI
scripts/           Index building, search, upload, ingestion, and evaluation CLIs
tests/             Unit tests for data, retrieval, API, security, and ingestion
evaluation/        Example and synthetic JSONL evaluation question sets
data/              Local source files, ignored for large PDFs
.rag_index/        Local generated vector index, ignored
uploads/           Runtime uploaded files, ignored
```

## Local Setup

Create the Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[mongodb]'
```

Install Strands only if you want the optional verification agent:

```bash
python -m pip install -e '.[mongodb,strands]'
```

Create `.env` from the example:

```bash
cp .env.example .env
```

Minimum local values:

```text
GOOGLE_API_KEY=your_gemini_key
RAG_API_KEYS=test-api-key
AUTH_MODE=api_key
CORS_ALLOWED_ORIGINS=http://127.0.0.1:3000,http://localhost:3000
```

For MongoDB Vector Search:

```text
VECTOR_BACKEND=mongodb
MONGODB_URI=mongodb+srv://...
MONGODB_DATABASE=healthcare_rag
MONGODB_COLLECTION=chunks
MONGODB_VECTOR_INDEX=chunk_embedding_vector_index
```

## Build The Initial Index

Place the DNV PDF or other source PDFs in `data/`.

Preview extraction and chunking:

```bash
python scripts/preview_data_layer.py --show 5
```

Build a local index:

```bash
python scripts/build_vector_index.py \
  --max-chapters 1000 \
  --chunk-words 900 \
  --overlap-words 150 \
  --batch-size 1
```

The local index is written to:

```text
.rag_index/embeddings.npz
.rag_index/metadata.json
```

Upload the local index to MongoDB:

```bash
python scripts/upload_vector_index_to_mongodb.py --index-dir .rag_index
```

Your MongoDB Atlas vector index must target the `embedding` field with:

```text
dimensions: 1024
similarity: cosine
```

Keep the Atlas index name equal to `MONGODB_VECTOR_INDEX`.

## Run FastAPI

Start the backend:

```bash
uvicorn rag_chatbot.api:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Interactive API docs:

```text
http://127.0.0.1:8000/docs
```

Ask a question:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-api-key" \
  -d '{
    "question": "What does IC.1 require?",
    "search_mode": "auto",
    "quality_mode": "fast",
    "top_k": 3,
    "rerank": false,
    "generate_answer": true
  }'
```

## Upload And Ask

The product flow is now one step from the user side:

```text
Upload document -> backend extracts, chunks, embeds, and saves to MongoDB -> ask
```

Use the combined endpoint:

```bash
curl -X POST http://127.0.0.1:8000/documents/upload-and-index \
  -H "X-API-Key: test-api-key" \
  -F "file=@data/sample.pdf"
```

The older debug endpoints still exist:

```bash
curl -X POST http://127.0.0.1:8000/documents/upload \
  -H "X-API-Key: test-api-key" \
  -F "file=@data/sample.pdf"

curl -X POST "http://127.0.0.1:8000/documents/<document_id>/ingest?show=5" \
  -H "X-API-Key: test-api-key"

curl -X POST "http://127.0.0.1:8000/documents/<document_id>/index" \
  -H "X-API-Key: test-api-key"
```

Uploaded files support:

```text
PDF
CSV
JSON
```

## Run The Frontend

The Next.js app lives in `frontend/`.

Create the frontend env file:

```bash
cd frontend
cp .env.example .env.local
```

Set:

```text
RAG_API_BASE_URL=http://127.0.0.1:8000
RAG_API_KEY=test-api-key
SESSION_SECRET=replace_with_a_long_random_session_secret
USERS_CSV_PATH=data/users.csv
```

Install and run:

```bash
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:3000
```

The frontend includes:

- signup and login
- server-side session cookie
- document upload
- automatic upload-and-index
- chat with citations
- source panels
- timings in seconds

For production-style local serving:

```bash
npm run build
npm run start -- --hostname 127.0.0.1 --port 3000
```

When using `next start`, rebuild after frontend code changes.

## Docker

Run the backend with Docker Compose:

```bash
docker compose up --build -d
```

Check the container:

```bash
docker ps
curl http://127.0.0.1:8000/health
```

Stop it:

```bash
docker compose down
```

The Compose file mounts `.rag_index` and a Hugging Face cache volume. For local
development it sets:

```text
PRELOAD_MODELS=false
```

That makes the API start faster. Models load lazily on the first retrieval,
upload-index, or rerank request.

Build manually:

```bash
docker build -t agentic-healthcare-rag:local .
```

Build with Strands:

```bash
docker build \
  --build-arg INSTALL_STRANDS=true \
  -t agentic-healthcare-rag:local .
```

## Retrieval Modes

The router can choose the retrieval mode automatically:

```text
keyword  -> exact codes and exact terms, e.g. IC.1 or QM.4
semantic -> natural-language meaning search
hybrid   -> combines keyword and semantic results
```

Hybrid search uses reciprocal-rank fusion so duplicate semantic and keyword
hits become one candidate instead of repeated output.

Default candidate depth:

```text
fast quality:
  hybrid 6, semantic 6, keyword 6, rerank off, verification off

balanced quality:
  keyword 5, semantic 8, hybrid 12, rerank on, verification on

strict quality:
  more candidates, rerank on, verification on
```

Use `fast` for local speed and `strict` when stronger evidence checking matters.

## Answer Format

Answers are structured for clinical reading:

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

Citation numbers map to the `sources` array returned by the API.

## Strands Verification

When enabled, the generated answer is sent to a Strands verification agent. The
agent checks whether each cited claim is supported by the retrieved source text.
Unsupported or unclear optional claims can be removed before the response is
returned.

Enable or disable:

```text
ENABLE_VERIFICATION=true
```

Verification metadata is returned with every answer:

```json
{
  "verification": {
    "enabled": true,
    "verified": true,
    "confidence": 1.0,
    "checked_claims": 4,
    "supported_claims": 4,
    "removed_claims": 0,
    "unclear_claims": 0
  }
}
```

## Security

Protected endpoints require `X-API-Key`.

```text
AUTH_MODE=api_key
RAG_API_KEYS=test-api-key
```

Security features:

- API key authentication
- constant-time API key comparison
- per-identity rate limiting
- prompt-injection detection before retrieval
- CORS allowlist
- unsafe answer filtering
- structured request logging with redaction

Rate limit settings:

```text
RATE_LIMIT_REQUESTS=5
RATE_LIMIT_WINDOW_SECONDS=60
```

For an AWS API Gateway or trusted reverse proxy:

```text
AUTH_MODE=trusted_proxy
TRUSTED_PROXY_SECRET=long_internal_secret
```

Do not expose the container directly to the internet in trusted proxy mode.

## Evaluation

Evaluation data lives in:

```text
evaluation/questions.example.jsonl
evaluation/synthetic_qa.jsonl
```

Run retrieval-only evaluation:

```bash
python scripts/evaluate_rag.py --no-rerank
```

Run answer generation evaluation:

```bash
python scripts/evaluate_rag.py --no-rerank --generate-answers
```

Run faithfulness judging:

```bash
python scripts/evaluate_rag.py \
  --no-rerank \
  --generate-answers \
  --evaluate-faithfulness
```

Evaluation metrics include:

- section hit rate
- mean reciprocal rank
- section recall
- expected-term recall
- answer-term recall
- citation validity
- fully supported claim rate
- grounded claim rate
- latency

Reports are written to:

```text
.rag_evaluation/report.json
```

Generated reports are ignored by Git.

## CI And Security Scans

GitHub Actions runs on pushes and pull requests.

CI checks:

- Python install
- source compilation
- unit tests
- Docker build

Security checks:

- Gitleaks secret scan
- dependency audit
- dependency review
- weekly scheduled scan

The workflows do not require the source PDF, `.rag_index`, or a Gemini key.

## AWS Notes

The project was prepared for container deployment, but local development is the
recommended path while experimenting because ECS/Fargate, CloudWatch, NAT, and
load balancers can create cost.

If deploying later:

- push the Docker image to ECR
- keep secrets in Secrets Manager
- use MongoDB Atlas for vector storage
- use API Gateway or an authenticated reverse proxy
- avoid public unauthenticated container access
- set billing alerts before starting ECS services

## Ignored Local Files

These are intentionally not committed:

```text
.env
.rag_index/
.rag_evaluation/
uploads/
data/*.pdf
frontend/.env.local
frontend/data/users.csv
frontend/.next/
frontend/node_modules/
```

## Models

The project uses:

- `BAAI/bge-m3` for embeddings
- `BAAI/bge-reranker-v2-m3` for reranking
- Gemini for routing and answer generation
- optional Strands agent for verification

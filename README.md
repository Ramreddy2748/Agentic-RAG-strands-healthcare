# Agentic RAG for Healthcare Accreditation

[![CI](https://github.com/Ramreddy2748/Agentic-RAG-strands-healthcare/actions/workflows/ci.yml/badge.svg)](https://github.com/Ramreddy2748/Agentic-RAG-strands-healthcare/actions/workflows/ci.yml)
[![Security](https://github.com/Ramreddy2748/Agentic-RAG-strands-healthcare/actions/workflows/security.yml/badge.svg)](https://github.com/Ramreddy2748/Agentic-RAG-strands-healthcare/actions/workflows/security.yml)

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
  -> Strands verification agent
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

Install the optional Strands agent dependency when running answer verification:

```bash
python -m pip install -e '.[strands]'
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
  -H "X-API-Key: $RAG_CLIENT_API_KEY" \
  -d '{
    "question": "What are the quality management responsibilities?",
    "search_mode": "auto",
    "top_k": 3,
    "rerank": true,
    "generate_answer": true
  }'
```

## Security Guardrails

The `/ask` endpoint requires an API key. Generate a strong key:

```bash
openssl rand -hex 32
```

Add it to `.env`:

```text
RAG_API_KEYS=generated_key_here
```

Set the same value in the client environment:

```bash
export RAG_CLIENT_API_KEY=generated_key_here
```

Multiple keys can be accepted during rotation:

```text
RAG_API_KEYS=current_key,next_key
```

API keys are compared in constant time and are never included in application
logs. If `RAG_API_KEYS` is missing, protected requests fail closed with `503`.
Missing or invalid client keys return `401`. The public `/health` endpoint
remains unauthenticated for container and load-balancer health checks.

Before routing or retrieval, the API locally rejects common prompt-injection
techniques, including trusted-instruction overrides, forged model roles,
system-prompt extraction, credential extraction, and jailbreak language.
Rejected questions return `prompt_injection_detected` and never enter the RAG
pipeline or make a Gemini call.

Each authenticated identity is also limited before expensive model work:

```text
RATE_LIMIT_REQUESTS=5
RATE_LIMIT_WINDOW_SECONDS=60
```

Exceeded limits return `429` with a `Retry-After` header. This in-process
limiter protects one API process. For multiple AWS tasks or replicas, configure
the same account-wide limit in API Gateway or AWS WAF.

Browser access is disabled unless exact origins are configured:

```text
CORS_ALLOWED_ORIGINS=https://app.example.org,https://admin.example.org
```

Wildcard origins are rejected. Restart the API after changing CORS settings.

After retrieval and reranking, the service checks the best evidence score:

```text
MIN_RERANK_SCORE=0.50
MIN_SEMANTIC_SCORE=0.35
MIN_KEYWORD_SCORE=0.01
```

If evidence is below the applicable threshold, Gemini answer generation is
skipped and the response reports `evidence_sufficient: false`. Tune thresholds
using the offline evaluation set before production deployment.

Generated answers are checked for prompt disclosure, credential language, and
configured secret values. Unsafe output is withheld. Structured logs recursively
redact questions, authorization fields, credentials, and known secret values
while retaining request IDs and non-sensitive timing data.

### AWS Authentication

For local development, use:

```text
AUTH_MODE=api_key
```

Behind an authenticated AWS API Gateway or trusted reverse proxy, use:

```text
AUTH_MODE=trusted_proxy
TRUSTED_PROXY_SECRET=long_internal_secret
```

The proxy must remove client-supplied `X-Authenticated-User` and
`X-Proxy-Secret` headers, authenticate the caller, then inject:

```text
X-Authenticated-User: stable-user-identity
X-Proxy-Secret: long_internal_secret
```

Do not expose the container directly to the internet in `trusted_proxy` mode.
Store `TRUSTED_PROXY_SECRET` and `GOOGLE_API_KEY` in AWS Secrets Manager rather
than the image or repository.

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

## Strands Verification Layer

When `ENABLE_VERIFICATION=true`, the generated clinical answer is passed to a
Strands verification agent before it is returned. The agent receives only:

```text
the user question
the generated answer claims
the retrieved source chunks cited by those claims
```

It checks each claim against its cited source text, marks claims as supported,
unsupported, or unclear, strips unsupported or unclear optional claims, and
returns verification metadata:

```json
{
  "verification": {
    "enabled": true,
    "verified": false,
    "confidence": 0.84,
    "checked_claims": 5,
    "supported_claims": 4,
    "removed_claims": 1,
    "unclear_claims": 0,
    "reason": "Removed 1 unsupported claim(s).",
    "unsupported_claims": ["Unsupported claim text"]
  }
}
```

This adds one extra agent/LLM call per generated answer. For retrieval-only or
low-cost local tests, set:

```text
ENABLE_VERIFICATION=false
```

For Docker builds that include Strands:

```bash
docker build --build-arg INSTALL_STRANDS=true -t agentic-healthcare-rag .
```

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
    "verification_ms": 920.0,
    "total_ms": 13641.28
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

## Offline Evaluation

Evaluation cases are stored as newline-delimited JSON. Each case can specify
expected sections and phrases:

```json
{"case_id":"ic-1","question":"What does IC.1 require?","expected_sections":["IC.1"],"expected_terms":["infection prevention and control program"],"search_mode":"hybrid"}
```

Run retrieval evaluation without Gemini answer generation:

```bash
python scripts/evaluate_rag.py --no-rerank
```

Run the complete pipeline, including reranking and grounded answers:

```bash
python scripts/evaluate_rag.py --generate-answers
```

Evaluate whether every answer claim is supported by its cited passages:

```bash
python scripts/evaluate_rag.py \
  --no-rerank \
  --generate-answers \
  --evaluate-faithfulness
```

Faithfulness mode uses one answer call and one judge call per question. A shared
limiter spaces all Gemini requests by at least 15 seconds, runs them
sequentially, and does not retry failed calls in application code. Cases must
use manual `semantic`, `keyword`, or `hybrid` modes so routing cannot add hidden
API requests. Change the interval only when the model's published rate limit
allows it:

```bash
python scripts/evaluate_rag.py \
  --no-rerank \
  --generate-answers \
  --evaluate-faithfulness \
  --api-interval-seconds 15
```

The command writes `.rag_evaluation/report.json` with section hit rate, mean
reciprocal rank, section recall, expected-term coverage, citation validity, and
latency. Faithfulness runs also report fully supported and grounded claim rates,
plus each unsupported claim and its reason. Failed questions are recorded
individually without stopping the batch. Generated reports remain local and are
excluded from Git.

## Docker

The image contains the API and Python dependencies. The generated vector index,
Gemini credentials, and Hugging Face model cache remain outside the image.

Build the image:

```bash
docker build -t agentic-healthcare-rag:local .
```

Build with the optional Strands verifier dependency:

```bash
docker build \
  --build-arg INSTALL_STRANDS=true \
  -t agentic-healthcare-rag:local .
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

## AWS Index Bootstrap

For ECS/Fargate, upload the generated index to S3:

```bash
aws s3 sync .rag_index s3://agentic-healthcare-rag-index/rag-index/ \
  --region us-east-1
```

The container checks `RAG_INDEX_DIR` during startup. If `metadata.json` and
`embeddings.npz` are missing locally and `INDEX_S3_BUCKET` is set, it downloads
both files from S3 before loading the RAG service.

Use these ECS environment variables:

```text
RAG_INDEX_DIR=/app/.rag_index
INDEX_S3_BUCKET=agentic-healthcare-rag-index
INDEX_S3_PREFIX=rag-index
AWS_REGION=us-east-1
```

The ECS task role needs:

```text
s3:GetObject on arn:aws:s3:::agentic-healthcare-rag-index/rag-index/*
```

Local development can keep using the mounted `.rag_index` directory and leave
`INDEX_S3_BUCKET` empty.

## Continuous Integration

GitHub Actions runs automatically on pushes and pull requests targeting `main`.

`CI` performs:

- Python 3.12 dependency installation
- Python source compilation
- All unit tests with model downloads and Gemini calls disabled
- A complete Docker image build without publishing the image

`Security` performs:

- Full-history secret scanning with Gitleaks
- Python dependency vulnerability auditing with `pip-audit`
- Dependency review for pull requests
- A scheduled security scan every Monday

View results under the repository's **Actions** tab. The workflows do not
require `GOOGLE_API_KEY`, the source PDF, or `.rag_index`.

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

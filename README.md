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
  -> local vector search
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

## Search the Vector Index

```bash
python scripts/search_vector_index.py \
  "What are the quality management responsibilities?"
```

## Embedding Model

The project uses the open-weight
[`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) model through
`sentence-transformers`.

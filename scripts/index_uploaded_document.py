from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_chatbot.embedding_layer import DEFAULT_MODEL_NAME
from rag_chatbot.indexing_layer import index_uploaded_document


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index one uploaded document into MongoDB Vector Search."
    )
    parser.add_argument("document_id")
    parser.add_argument("--upload-dir", default="uploads")
    parser.add_argument("--chunk-words", type=int, default=900)
    parser.add_argument("--overlap-words", type=int, default=150)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--mongo-batch-size", type=int, default=100)
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()

    load_dotenv()
    result = index_uploaded_document(
        args.document_id,
        upload_dir=args.upload_dir,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
        model_name=args.model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_NAME),
        embedding_batch_size=args.embedding_batch_size,
        mongo_batch_size=args.mongo_batch_size,
    )
    print(f"Document: {result.filename}")
    print(f"Elements: {result.element_count}")
    print(f"Chunks: {result.chunk_count}")
    print(f"Upserted: {result.upserted_count}")
    print(f"Embedding model: {result.model_name}")


if __name__ == "__main__":
    main()

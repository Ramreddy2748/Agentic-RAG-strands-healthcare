from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_chatbot.embedding_layer import DEFAULT_INDEX_DIR, load_vector_index
from rag_chatbot.mongo_vector_store import MongoVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload a local .rag_index into MongoDB Atlas Vector Search."
    )
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    load_dotenv()
    index = load_vector_index(args.index_dir)
    store = MongoVectorStore.from_env()
    written = store.upsert_vector_index(index, batch_size=args.batch_size)

    print(f"MongoDB database: {store.config.database}")
    print(f"MongoDB collection: {store.config.collection}")
    print(f"MongoDB vector index: {store.config.vector_index}")
    print(f"Chunks in local index: {len(index.chunks)}")
    print(f"Chunks uploaded or updated: {written}")


if __name__ == "__main__":
    main()

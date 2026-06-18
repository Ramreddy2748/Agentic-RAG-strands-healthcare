from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_chatbot.embedding_layer import (
    DEFAULT_INDEX_DIR,
    load_vector_index,
    search_vector_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the local vector index.")
    parser.add_argument("query")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    index = load_vector_index(args.index_dir)
    results = search_vector_index(
        index,
        args.query,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )

    for number, result in enumerate(results, start=1):
        chunk = result.chunk
        page_range = str(chunk.page_number)
        if chunk.end_page_number != chunk.page_number:
            page_range = f"{chunk.page_number}-{chunk.end_page_number}"

        print()
        print(f"[{number}] score={result.score:.4f} pages={page_range}")
        print(f"Chapter: {chunk.chapter_title}")
        print(f"Section: {chunk.section_title}")
        print(chunk.text[:900])


if __name__ == "__main__":
    main()

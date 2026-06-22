from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_chatbot.data_layer import chunk_sections, load_pages, pages_to_sections
from rag_chatbot.embedding_layer import (
    DEFAULT_INDEX_DIR,
    DEFAULT_MODEL_NAME,
    build_vector_index,
    save_vector_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local BGE-M3 vector index.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--chunk-words", type=int, default=900)
    parser.add_argument("--overlap-words", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-chapters",
        type=int,
        default=3,
        help="Embed only the first N chapters. Default: 3.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show selected chapters and chunks without creating embeddings.",
    )
    parser.add_argument("--include-front-matter", action="store_true")
    args = parser.parse_args()

    pages = load_pages(args.data_dir)
    sections = pages_to_sections(pages)
    if not args.include_front_matter:
        sections = [
            section
            for section in sections
            if section.chapter_title != "Front Matter"
        ]

    chapter_names = list(dict.fromkeys(section.chapter_title for section in sections))
    selected_chapters = chapter_names[: args.max_chapters]
    selected_sections = [
        section
        for section in sections
        if section.chapter_title in selected_chapters
    ]
    chunks = chunk_sections(
        selected_sections,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
    )

    print("Selected chapters:")
    for chapter in selected_chapters:
        print(f"- {chapter}")
    print(f"Chunks to embed: {len(chunks)}")
    print(f"Embedding model: {args.model_name}")
    if args.dry_run:
        print("Dry run complete. No embeddings were created or saved.")
        return

    index = build_vector_index(
        chunks,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )
    save_vector_index(index, args.index_dir)
    print(f"Saved vector index to: {args.index_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_chatbot.data_layer import chunk_pages, load_pages, pages_to_sections


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview Data Layer extraction.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--chunk-words",
        "--chunk-size",
        dest="chunk_words",
        type=int,
        default=900,
    )
    parser.add_argument(
        "--overlap-words",
        "--overlap",
        dest="overlap_words",
        type=int,
        default=150,
    )
    parser.add_argument("--include-front-matter", action="store_true")
    parser.add_argument("--show", type=int, default=3)
    args = parser.parse_args()

    pages = load_pages(args.data_dir)
    sections = pages_to_sections(pages)
    chunks = chunk_pages(
        pages,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
        include_front_matter=args.include_front_matter,
    )

    print(f"Pages extracted: {len(pages)}")
    print(f"Sections detected: {len(sections)}")
    if not args.include_front_matter:
        print("Front matter chunks: skipped")
    print(f"Chunks created: {len(chunks)}")
    for chunk in chunks[: args.show]:
        print()
        page_range = f"page {chunk.page_number}"
        if chunk.end_page_number != chunk.page_number:
            page_range = f"pages {chunk.page_number}-{chunk.end_page_number}"
        print(f"{chunk.chunk_id} | {page_range} | {chunk.word_count} words")
        print(f"Chapter: {chunk.chapter_title}")
        print(f"Section: {chunk.section_title}")
        print(chunk.text[:500])


if __name__ == "__main__":
    main()

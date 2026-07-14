from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_chatbot.ingestion_layer import ingest_uploaded_document


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview extraction for one uploaded document."
    )
    parser.add_argument("document_id")
    parser.add_argument("--upload-dir", default="uploads")
    parser.add_argument("--show", type=int, default=5)
    args = parser.parse_args()

    result = ingest_uploaded_document(args.document_id, upload_dir=args.upload_dir)
    print(f"Document: {result.filename}")
    print(f"Type: {result.file_extension}")
    print(f"Elements: {result.element_count}")
    for index, element in enumerate(result.elements[: args.show], start=1):
        location = ""
        if element.page_number is not None:
            location = f" page={element.page_number}"
        elif element.row_number is not None:
            location = f" row={element.row_number}"
        elif element.json_path is not None:
            location = f" path={element.json_path}"
        print(f"\n[{index}] {element.content_type}{location}")
        print(element.text[:1000])


if __name__ == "__main__":
    main()

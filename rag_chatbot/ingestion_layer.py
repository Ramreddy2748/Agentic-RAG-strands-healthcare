from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from rag_chatbot.data_layer import (
    DocumentSource,
    extract_pdf_pages,
    normalize_text,
)
from rag_chatbot.document_layer import UploadedDocument, load_document_metadata


@dataclass(frozen=True)
class DocumentElement:
    """Normalized content extracted from any supported source file."""

    source_id: str
    source_path: str
    content_type: str
    text: str
    page_number: int | None = None
    row_number: int | None = None
    json_path: str | None = None
    metadata: dict[str, str] | None = None


@dataclass(frozen=True)
class IngestionResult:
    """Extraction result for one uploaded document."""

    document_id: str
    filename: str
    file_extension: str
    element_count: int
    elements: list[DocumentElement]


def ingest_uploaded_document(
    document_id: str,
    *,
    upload_dir: str | Path = "uploads",
) -> IngestionResult:
    """Extract normalized elements for one uploaded document."""
    documents = load_document_metadata(upload_dir)
    matches = [item for item in documents if item.document_id == document_id]
    if not matches:
        raise FileNotFoundError(f"Uploaded document not found: {document_id}")
    return ingest_document(matches[0])


def ingest_document(document: UploadedDocument) -> IngestionResult:
    """Dispatch one uploaded document to the correct extractor."""
    extension = f".{document.file_extension.lower()}"
    path = Path(document.stored_path)
    if not path.exists():
        raise FileNotFoundError(f"Uploaded file not found: {path}")

    if extension == ".pdf":
        elements = extract_pdf_elements(document)
    elif extension == ".csv":
        elements = extract_csv_elements(document)
    elif extension == ".json":
        elements = extract_json_elements(document)
    else:
        raise ValueError(f"Unsupported uploaded document type: {extension}")

    return IngestionResult(
        document_id=document.document_id,
        filename=document.original_filename,
        file_extension=document.file_extension,
        element_count=len(elements),
        elements=elements,
    )


def extract_pdf_elements(document: UploadedDocument) -> list[DocumentElement]:
    """Extract one normalized text element per readable PDF page."""
    source = DocumentSource(path=Path(document.stored_path))
    pages = extract_pdf_pages(source)
    return [
        DocumentElement(
            source_id=document.document_id,
            source_path=document.stored_path,
            content_type="pdf_page",
            page_number=page.page_number,
            text=page.text,
            metadata={
                "original_filename": document.original_filename,
                "pdf_source_id": page.source_id,
            },
        )
        for page in pages
    ]


def extract_csv_elements(document: UploadedDocument) -> list[DocumentElement]:
    """Extract one readable text element per CSV row."""
    path = Path(document.stored_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample) if sample.strip() else csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        if not reader.fieldnames:
            return []
        rows = list(reader)

    elements: list[DocumentElement] = []
    for row_number, row in enumerate(rows, start=1):
        parts = [
            f"{key}: {normalize_text(value)}"
            for key, value in row.items()
            if key and value and normalize_text(value)
        ]
        if not parts:
            continue
        elements.append(
            DocumentElement(
                source_id=document.document_id,
                source_path=document.stored_path,
                content_type="csv_row",
                row_number=row_number,
                text="\n".join(parts),
                metadata={
                    "original_filename": document.original_filename,
                    "columns": ", ".join(reader.fieldnames or []),
                },
            )
        )
    return elements


def extract_json_elements(document: UploadedDocument) -> list[DocumentElement]:
    """Extract readable text elements from JSON leaf paths."""
    path = Path(document.stored_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    elements: list[DocumentElement] = []
    for json_path, value in flatten_json(data):
        text = normalize_text(str(value))
        if not text:
            continue
        elements.append(
            DocumentElement(
                source_id=document.document_id,
                source_path=document.stored_path,
                content_type="json_value",
                json_path=json_path,
                text=f"{json_path}: {text}",
                metadata={"original_filename": document.original_filename},
            )
        )
    return elements


def flatten_json(value: Any, prefix: str = "$") -> list[tuple[str, Any]]:
    """Return leaf JSON paths and scalar values."""
    if isinstance(value, dict):
        items: list[tuple[str, Any]] = []
        for key, child in value.items():
            child_path = f"{prefix}.{key}" if prefix else str(key)
            items.extend(flatten_json(child, child_path))
        return items
    if isinstance(value, list):
        items = []
        for index, child in enumerate(value):
            items.extend(flatten_json(child, f"{prefix}[{index}]"))
        return items
    if value is None:
        return []
    return [(prefix, value)]

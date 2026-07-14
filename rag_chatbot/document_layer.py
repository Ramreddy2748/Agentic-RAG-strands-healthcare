from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from uuid import uuid4


DEFAULT_UPLOAD_DIR = Path("uploads")
DOCUMENTS_FILE = "documents.json"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".json"}
SUPPORTED_CONTENT_TYPES = {
    "application/pdf",
    "text/csv",
    "application/csv",
    "application/json",
    "text/json",
}


@dataclass(frozen=True)
class UploadedDocument:
    """Metadata for one uploaded document before ingestion."""

    document_id: str
    original_filename: str
    stored_filename: str
    stored_path: str
    content_type: str
    file_extension: str
    size_bytes: int
    status: str
    created_at: str


def save_uploaded_document(
    *,
    filename: str,
    content_type: str | None,
    content: bytes,
    upload_dir: str | Path = DEFAULT_UPLOAD_DIR,
) -> UploadedDocument:
    """Validate and persist one uploaded source document."""
    if not filename.strip():
        raise ValueError("Uploaded file must have a filename.")
    if not content:
        raise ValueError("Uploaded file cannot be empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"Uploaded file is too large. Maximum size is {MAX_UPLOAD_BYTES} bytes."
        )

    safe_name = sanitize_filename(filename)
    extension = Path(safe_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Only PDF, CSV, and JSON uploads are supported.")

    normalized_content_type = (content_type or "").split(";")[0].strip().lower()
    if normalized_content_type and normalized_content_type not in SUPPORTED_CONTENT_TYPES:
        raise ValueError(f"Unsupported content type: {normalized_content_type}")

    document_id = uuid4().hex
    stored_filename = f"{document_id}{extension}"
    root = Path(upload_dir)
    root.mkdir(parents=True, exist_ok=True)
    stored_path = root / stored_filename
    stored_path.write_bytes(content)

    document = UploadedDocument(
        document_id=document_id,
        original_filename=safe_name,
        stored_filename=stored_filename,
        stored_path=str(stored_path),
        content_type=normalized_content_type or infer_content_type(extension),
        file_extension=extension.lstrip("."),
        size_bytes=len(content),
        status="uploaded",
        created_at=datetime.now(UTC).isoformat(),
    )
    append_document_metadata(root, document)
    return document


def sanitize_filename(filename: str) -> str:
    """Return a basename safe enough for metadata display and extension checks."""
    basename = Path(filename).name.strip()
    return re.sub(r"[^A-Za-z0-9._ -]", "_", basename)


def infer_content_type(extension: str) -> str:
    """Infer content type from a validated file extension."""
    if extension == ".pdf":
        return "application/pdf"
    if extension == ".csv":
        return "text/csv"
    return "application/json"


def append_document_metadata(upload_dir: Path, document: UploadedDocument) -> None:
    """Append one document record to the upload metadata file."""
    metadata_path = upload_dir / DOCUMENTS_FILE
    documents = load_document_metadata(upload_dir)
    documents.append(document)
    metadata_path.write_text(
        json.dumps([asdict(item) for item in documents], indent=2),
        encoding="utf-8",
    )


def load_document_metadata(
    upload_dir: str | Path = DEFAULT_UPLOAD_DIR,
) -> list[UploadedDocument]:
    """Load uploaded document metadata from local storage."""
    metadata_path = Path(upload_dir) / DOCUMENTS_FILE
    if not metadata_path.exists():
        return []
    raw_documents = json.loads(metadata_path.read_text(encoding="utf-8"))
    return [UploadedDocument(**item) for item in raw_documents]

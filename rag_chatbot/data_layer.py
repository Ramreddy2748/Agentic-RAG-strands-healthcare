from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
from typing import Iterable


DEFAULT_DATA_DIR = Path("data")
SUPPORTED_EXTENSIONS = {".pdf"}


@dataclass(frozen=True)
class DocumentSource:
    """A source file available for ingestion."""

    path: Path

    @property
    def source_id(self) -> str:
        return self.path.stem # here we are sending pdf name to chunk_id


@dataclass(frozen=True)
class DocumentPage:
    """Text extracted from one document page."""

    source_id: str
    source_path: Path
    page_number: int
    text: str


@dataclass(frozen=True)
class DocumentSection:
    """A logical document section assembled from one or more pages."""

    section_id: str
    source_id: str
    source_path: Path
    chapter_title: str
    section_title: str
    start_page: int
    end_page: int
    text: str


@dataclass(frozen=True)
class DocumentChunk:
    """A searchable unit of document text with source metadata."""

    chunk_id: str
    source_id: str
    source_path: Path
    page_number: int
    end_page_number: int
    chapter_title: str
    section_title: str
    text: str
    word_count: int


def discover_sources(data_dir: str | Path = DEFAULT_DATA_DIR) -> list[DocumentSource]:
    """Return supported document sources from the data directory."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {root}")

    sources = [
        DocumentSource(path=path)
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sources


def extract_pdf_pages(source: DocumentSource) -> list[DocumentPage]:
    """Extract page text from a PDF source."""
    if shutil.which("pdftotext"):
        return extract_pdf_pages_with_pdftotext(source)

    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing PDF parser dependency. Install project dependencies with "
            "`python3 -m pip install -e .`, or install the `pdftotext` command."
        ) from exc

    reader = PdfReader(str(source.path))
    pages: list[DocumentPage] = []
    for index, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if not text:
            continue
        pages.append(
            DocumentPage(
                source_id=source.source_id,
                source_path=source.path,
                page_number=index,
                text=text,
            )
        )
    return pages



def extract_pdf_pages_with_pdftotext(source: DocumentSource) -> list[DocumentPage]:
    """Extract page text through the local pdftotext command."""
    result = subprocess.run(
        ["pdftotext", "-layout", str(source.path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    raw_pages = result.stdout.split("\f")
    pages: list[DocumentPage] = []
    for index, raw_text in enumerate(raw_pages, start=1):
        text = normalize_text(raw_text)
        if not text:
            continue
        pages.append(
            DocumentPage(
                source_id=source.source_id,
                source_path=source.path,
                page_number=index,
                text=text,
            )
        )
    return pages



def load_pages(data_dir: str | Path = DEFAULT_DATA_DIR) -> list[DocumentPage]:
    """Discover and extract all supported documents from the data directory."""
    pages: list[DocumentPage] = []
    for source in discover_sources(data_dir):
        pages.extend(extract_pdf_pages(source))
    return pages



def chunk_pages(
    pages: Iterable[DocumentPage],
    *,
    chunk_words: int = 900,
    overlap_words: int = 150,
    include_front_matter: bool = False,
) -> list[DocumentChunk]:
    """Split pages into section-aware overlapping word chunks."""
    sections = pages_to_sections(pages)
    if not include_front_matter:
        sections = [
            section
            for section in sections
            if section.chapter_title != "Front Matter"
        ]
    return chunk_sections(
        sections,
        chunk_words=chunk_words,
        overlap_words=overlap_words,
    )


def pages_to_sections(pages: Iterable[DocumentPage]) -> list[DocumentSection]:
    """Group page text into logical sections using visible PDF headings."""
    sections: list[DocumentSection] = []
    current_source_id = ""
    current_source_path = Path()
    current_chapter = "Front Matter"
    current_title = "Front Matter"
    current_start_page = 0
    current_end_page = 0
    current_parts: list[str] = []

    def flush_section() -> None:
        nonlocal current_parts
        if not current_parts:
            return
        text = "\n\n".join(current_parts).strip()
        if not text:
            current_parts = []
            return
        section_slug = slugify(current_title)
        sections.append(
            DocumentSection(
                section_id=(
                    # here we are adding all rhe source_id(pdf_name, start page, end page)
                    f"{current_source_id}:p{current_start_page}-"
                    f"{current_end_page}:{section_slug}"
                ),
                source_id=current_source_id,
                source_path=current_source_path,
                chapter_title=current_chapter,
                section_title=current_title,
                start_page=current_start_page,
                end_page=current_end_page,
                text=text,
            )
        )
        current_parts = []

    for page in pages:
        paragraphs = [part.strip() for part in page.text.split("\n\n") if part.strip()]
        for paragraph in paragraphs:
            if is_repeated_page_chrome(paragraph):
                continue

            if is_chapter_heading(paragraph):
                flush_section()
                current_source_id = page.source_id
                current_source_path = page.source_path
                current_chapter = paragraph
                current_title = paragraph
                current_start_page = page.page_number
                current_end_page = page.page_number
                current_parts = []
                continue

            if is_section_heading(paragraph):
                flush_section()
                current_source_id = page.source_id
                current_source_path = page.source_path
                current_title = paragraph
                current_start_page = page.page_number
                current_end_page = page.page_number
                current_parts = [paragraph]
                continue

            if not current_parts:
                current_source_id = page.source_id
                current_source_path = page.source_path
                current_start_page = page.page_number
                current_title = current_chapter

            current_end_page = page.page_number
            current_parts.append(paragraph)

    flush_section()
    return sections


def chunk_sections(
    sections: Iterable[DocumentSection],
    *,
    chunk_words: int = 900,
    overlap_words: int = 150,
) -> list[DocumentChunk]:
    """Split document sections into deterministic overlapping word chunks."""
    if chunk_words < 100:
        raise ValueError("chunk_words must be at least 100 words")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    chunks: list[DocumentChunk] = []
    for section in sections:
        section_chunks = split_words(
            section.text,
            chunk_words=chunk_words,
            overlap_words=overlap_words,
        )
        for index, text in enumerate(section_chunks, start=1):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{section.section_id}:c{index}",
                    source_id=section.source_id,
                    source_path=section.source_path,
                    page_number=section.start_page,
                    end_page_number=section.end_page,
                    chapter_title=section.chapter_title,
                    section_title=section.section_title,
                    text=text,
                    word_count=len(text.split()),
                )
            )
    return chunks


def split_words(text: str, *, chunk_words: int, overlap_words: int) -> list[str]:
    """Split text into word-count chunks with overlap."""
    words = normalize_text(text).split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = max(end - overlap_words, start + 1)
    return chunks



def is_chapter_heading(text: str) -> bool:
    """Return whether text looks like a top-level NIAHO chapter heading."""
    value = text.strip()
    if len(value) > 90 or len(value.split()) > 8:
        return False
    if not re.search(r"\([A-Z]{2,6}\)$", value):
        return False
    return value == value.upper()


def is_section_heading(text: str) -> bool:
    """Return whether text looks like a numbered NIAHO section heading."""
    value = text.strip()
    if len(value) > 140 or "..." in value:
        return False

    match = re.match(r"^([A-Z]{1,6})\.\d+[A-Z]?(?:\.\d+)?\s+\S+", value)
    if not match:
        return False
    return match.group(1) != "SR"


def is_repeated_page_chrome(text: str) -> bool:
    """Skip repeated headers and footers from extracted PDF pages."""
    value = text.strip()
    if re.match(r"^Page \d+ of \d+$", value):
        return True
    if "NIAHO Accreditation Requirements" in value and "Revision 25-1" in value:
        return True
    return False


def slugify(text: str) -> str:
    """Make stable, readable IDs from section titles."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:80] or "section"


# Normalization like removing spaces between pages or extra spaces.

def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks."""
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    paragraphs: list[str] = []
    current: list[str] = []

    for line in lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(" ".join(line.split()))

    if current:
        paragraphs.append(" ".join(current))

    return "\n\n".join(paragraphs).strip()

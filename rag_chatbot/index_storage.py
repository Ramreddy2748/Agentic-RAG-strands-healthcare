from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Protocol


INDEX_FILES = ("metadata.json", "embeddings.npz")


class S3Client(Protocol):
    """Small subset of the boto3 S3 client used by index bootstrap."""

    def download_file(self, bucket: str, key: str, filename: str) -> None: ...


@dataclass(frozen=True)
class IndexBootstrapResult:
    """Result of checking or downloading a persisted RAG index."""

    index_dir: Path
    available: bool
    downloaded: bool
    source: str


def index_is_available(index_dir: str | Path) -> bool:
    """Return whether the expected persisted index files exist locally."""
    path = Path(index_dir)
    return all((path / filename).exists() for filename in INDEX_FILES)


def ensure_index_available(
    index_dir: str | Path,
    *,
    bucket: str | None = None,
    prefix: str | None = None,
    region: str | None = None,
    s3_client: S3Client | None = None,
) -> IndexBootstrapResult:
    """Ensure the local index exists, downloading it from S3 when configured."""
    path = Path(index_dir)
    if index_is_available(path):
        return IndexBootstrapResult(
            index_dir=path,
            available=True,
            downloaded=False,
            source="local",
        )

    resolved_bucket = bucket or os.getenv("INDEX_S3_BUCKET")
    if not resolved_bucket:
        return IndexBootstrapResult(
            index_dir=path,
            available=False,
            downloaded=False,
            source="missing",
        )

    resolved_prefix = prefix if prefix is not None else os.getenv("INDEX_S3_PREFIX", "")
    active_client = s3_client or build_s3_client(region)
    download_index_from_s3(
        path,
        bucket=resolved_bucket,
        prefix=resolved_prefix,
        s3_client=active_client,
    )
    return IndexBootstrapResult(
        index_dir=path,
        available=index_is_available(path),
        downloaded=True,
        source=f"s3://{resolved_bucket}/{normalize_prefix(resolved_prefix)}",
    )


def build_s3_client(region: str | None = None) -> S3Client:
    """Create a boto3 S3 client only when S3 bootstrap is actually needed."""
    try:
        import boto3
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing AWS dependency. Install project dependencies with "
            "`python -m pip install -e .` before using S3 index bootstrap."
        ) from exc

    client_kwargs = {}
    resolved_region = region or os.getenv("AWS_REGION")
    if resolved_region:
        client_kwargs["region_name"] = resolved_region
    return boto3.client("s3", **client_kwargs)


def download_index_from_s3(
    index_dir: Path,
    *,
    bucket: str,
    prefix: str,
    s3_client: S3Client,
) -> None:
    """Download the required persisted index files from one S3 prefix."""
    index_dir.mkdir(parents=True, exist_ok=True)
    normalized_prefix = normalize_prefix(prefix)
    for filename in INDEX_FILES:
        key = s3_key(normalized_prefix, filename)
        s3_client.download_file(bucket, key, str(index_dir / filename))


def normalize_prefix(prefix: str | None) -> str:
    """Normalize an S3 prefix for joining with object names."""
    return (prefix or "").strip("/")


def s3_key(prefix: str, filename: str) -> str:
    """Return an S3 object key for a file below an optional prefix."""
    if not prefix:
        return filename
    return f"{prefix}/{filename}"

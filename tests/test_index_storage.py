from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rag_chatbot.index_storage import (
    INDEX_FILES,
    ensure_index_available,
    index_is_available,
    normalize_prefix,
    s3_key,
)


class FakeS3Client:
    def __init__(self) -> None:
        self.downloads: list[tuple[str, str, str]] = []

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        self.downloads.append((bucket, key, filename))
        Path(filename).write_text(f"downloaded {key}", encoding="utf-8")


class IndexStorageTests(unittest.TestCase):
    def test_index_is_available_when_required_files_exist(self) -> None:
        with TemporaryDirectory() as temp_dir:
            index_dir = Path(temp_dir)
            for filename in INDEX_FILES:
                (index_dir / filename).write_text("x", encoding="utf-8")

            self.assertTrue(index_is_available(index_dir))

    def test_missing_local_index_without_bucket_returns_unavailable(self) -> None:
        with TemporaryDirectory() as temp_dir:
            result = ensure_index_available(temp_dir, bucket="")

            self.assertFalse(result.available)
            self.assertFalse(result.downloaded)
            self.assertEqual(result.source, "missing")

    def test_downloads_required_files_from_s3_prefix(self) -> None:
        with TemporaryDirectory() as temp_dir:
            client = FakeS3Client()
            result = ensure_index_available(
                temp_dir,
                bucket="rag-index-bucket",
                prefix="/rag-index/",
                s3_client=client,
            )

            self.assertTrue(result.available)
            self.assertTrue(result.downloaded)
            self.assertEqual(
                [download[1] for download in client.downloads],
                ["rag-index/metadata.json", "rag-index/embeddings.npz"],
            )

    def test_existing_local_index_skips_s3_download(self) -> None:
        with TemporaryDirectory() as temp_dir:
            index_dir = Path(temp_dir)
            for filename in INDEX_FILES:
                (index_dir / filename).write_text("x", encoding="utf-8")
            client = FakeS3Client()

            result = ensure_index_available(
                temp_dir,
                bucket="rag-index-bucket",
                prefix="rag-index",
                s3_client=client,
            )

            self.assertTrue(result.available)
            self.assertFalse(result.downloaded)
            self.assertEqual(client.downloads, [])

    def test_s3_key_normalizes_empty_and_nested_prefixes(self) -> None:
        self.assertEqual(normalize_prefix("/rag-index/"), "rag-index")
        self.assertEqual(s3_key("", "metadata.json"), "metadata.json")
        self.assertEqual(
            s3_key("rag-index", "metadata.json"),
            "rag-index/metadata.json",
        )


if __name__ == "__main__":
    unittest.main()

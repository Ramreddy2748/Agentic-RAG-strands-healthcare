from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_chatbot.document_layer import (
    load_document_metadata,
    sanitize_filename,
    save_uploaded_document,
)


class DocumentLayerTests(unittest.TestCase):
    def test_save_uploaded_document_persists_file_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            document = save_uploaded_document(
                filename="policy.pdf",
                content_type="application/pdf",
                content=b"%PDF-1.4 test",
                upload_dir=temp_dir,
            )

            self.assertEqual(document.original_filename, "policy.pdf")
            self.assertEqual(document.file_extension, "pdf")
            self.assertEqual(document.status, "uploaded")
            self.assertEqual(document.size_bytes, len(b"%PDF-1.4 test"))
            self.assertTrue(Path(document.stored_path).exists())

            documents = load_document_metadata(temp_dir)
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].document_id, document.document_id)

    def test_save_uploaded_document_rejects_unsupported_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                save_uploaded_document(
                    filename="notes.txt",
                    content_type="text/plain",
                    content=b"hello",
                    upload_dir=temp_dir,
                )

    def test_sanitize_filename_removes_path_and_unsafe_characters(self) -> None:
        self.assertEqual(
            sanitize_filename("../bad/name?.pdf"),
            "name_.pdf",
        )


if __name__ == "__main__":
    unittest.main()

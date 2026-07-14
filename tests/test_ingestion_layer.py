from __future__ import annotations

import tempfile
import unittest

from rag_chatbot.document_layer import save_uploaded_document
from rag_chatbot.ingestion_layer import (
    extract_csv_elements,
    extract_json_elements,
    flatten_json,
    ingest_uploaded_document,
)


class IngestionLayerTests(unittest.TestCase):
    def test_extract_csv_elements_creates_row_elements(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            document = save_uploaded_document(
                filename="policies.csv",
                content_type="text/csv",
                content=b"section,requirement\nIC.1,Maintain IPCP\nQM.1,Review QMS\n",
                upload_dir=temp_dir,
            )

            elements = extract_csv_elements(document)

            self.assertEqual(len(elements), 2)
            self.assertEqual(elements[0].content_type, "csv_row")
            self.assertEqual(elements[0].row_number, 1)
            self.assertIn("section: IC.1", elements[0].text)
            self.assertIn("requirement: Maintain IPCP", elements[0].text)

    def test_extract_json_elements_creates_path_elements(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            document = save_uploaded_document(
                filename="policy.json",
                content_type="application/json",
                content=b'{"section": "IC.1", "items": [{"text": "Maintain IPCP"}]}',
                upload_dir=temp_dir,
            )

            elements = extract_json_elements(document)

            paths = {element.json_path for element in elements}
            self.assertIn("$.section", paths)
            self.assertIn("$.items[0].text", paths)

    def test_ingest_uploaded_document_finds_metadata_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            document = save_uploaded_document(
                filename="policies.csv",
                content_type="text/csv",
                content=b"section,requirement\nIC.1,Maintain IPCP\n",
                upload_dir=temp_dir,
            )

            result = ingest_uploaded_document(document.document_id, upload_dir=temp_dir)

            self.assertEqual(result.document_id, document.document_id)
            self.assertEqual(result.filename, "policies.csv")
            self.assertEqual(result.element_count, 1)

    def test_flatten_json_skips_nulls_and_keeps_leaf_paths(self) -> None:
        flattened = flatten_json({"a": [1, None, {"b": "value"}]})

        self.assertEqual(flattened, [("$.a[0]", 1), ("$.a[2].b", "value")])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from rag_chatbot.keyword_layer import BM25Index, tokenize, tokenize_query

from test_reranking_layer import make_chunk


class KeywordLayerTests(unittest.TestCase):
    def test_tokenizer_preserves_requirement_codes(self) -> None:
        self.assertEqual(tokenize("QM.1 and SR.1a QAPI"), ["qm.1", "and", "sr.1a", "qapi"])

    def test_query_tokenizer_normalizes_compact_codes(self) -> None:
        self.assertEqual(tokenize_query("what is QM1"), ["qm.1"])
        self.assertEqual(tokenize_query("Explain SR1a QAPI"), ["sr.1a", "qapi"])

    def test_bm25_prefers_exact_keyword_match(self) -> None:
        chunks = [make_chunk(index) for index in range(3)]
        chunks[2] = chunks[2].__class__(
            **{
                **chunks[2].__dict__,
                "text": "The QAPI program follows requirement SR.1a.",
            }
        )

        results = BM25Index(chunks).search("SR.1a QAPI", top_k=3)

        self.assertEqual(results[0].chunk.chunk_id, "c2")
        self.assertGreater(results[0].score, 0)

    def test_exact_section_code_prefers_opening_chunk(self) -> None:
        chunks = [make_chunk(index) for index in range(2)]
        chunks[0] = chunks[0].__class__(
            **{
                **chunks[0].__dict__,
                "section_title": "QM.1 RESPONSIBILITY AND ACCOUNTABILITY",
                "text": "QM.1 RESPONSIBILITY AND ACCOUNTABILITY begins here.",
            }
        )
        chunks[1] = chunks[1].__class__(
            **{
                **chunks[1].__dict__,
                "section_title": "QM.1 RESPONSIBILITY AND ACCOUNTABILITY",
                "text": "Later guidance refers to QM.1.",
            }
        )

        results = BM25Index(chunks).search("what is QM1", top_k=2)

        self.assertEqual(results[0].chunk.chunk_id, "c0")


if __name__ == "__main__":
    unittest.main()

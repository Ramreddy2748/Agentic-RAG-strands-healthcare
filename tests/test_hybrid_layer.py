from __future__ import annotations

import unittest

from rag_chatbot.embedding_layer import SearchResult
from rag_chatbot.hybrid_layer import fuse_search_results
from rag_chatbot.keyword_layer import KeywordSearchResult

from test_reranking_layer import make_chunk


class HybridLayerTests(unittest.TestCase):
    def test_fusion_rewards_chunks_found_by_both_searches(self) -> None:
        semantic = [
            SearchResult(score=0.9, chunk=make_chunk(0)),
            SearchResult(score=0.8, chunk=make_chunk(1)),
        ]
        keyword = [
            KeywordSearchResult(score=4.2, chunk=make_chunk(1)),
            KeywordSearchResult(score=3.1, chunk=make_chunk(2)),
        ]

        results = fuse_search_results(semantic, keyword, top_k=3)

        self.assertEqual(results[0].chunk.chunk_id, "c1")
        self.assertEqual(results[0].vector_score, 0.8)
        self.assertEqual(results[0].keyword_score, 4.2)


if __name__ == "__main__":
    unittest.main()

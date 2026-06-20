from __future__ import annotations

import unittest

from rag_chatbot.embedding_layer import IndexedChunk, SearchResult
from rag_chatbot.reranking_layer import rerank_search_results


class FakeReranker:
    def score(
        self,
        query: str,
        passages: list[str],
        *,
        batch_size: int = 2,
    ) -> list[float]:
        return [0.2, 0.9, 0.5]


class RerankingLayerTests(unittest.TestCase):
    def test_reranker_reorders_vector_candidates(self) -> None:
        candidates = [
            SearchResult(score=0.9 - index * 0.1, chunk=make_chunk(index))
            for index in range(3)
        ]

        results = rerank_search_results(
            "quality responsibilities",
            candidates,
            top_k=2,
            reranker=FakeReranker(),
        )

        self.assertEqual([result.chunk.chunk_id for result in results], ["c1", "c2"])
        self.assertEqual(results[0].rerank_score, 0.9)
        self.assertEqual(results[0].vector_score, 0.8)
        self.assertEqual(results[0].keyword_score, 0.0)


def make_chunk(index: int) -> IndexedChunk:
    return IndexedChunk(
        chunk_id=f"c{index}",
        source_id="source",
        source_path="data/source.pdf",
        page_number=index + 1,
        end_page_number=index + 1,
        chapter_title="QUALITY MANAGEMENT SYSTEM (QM)",
        section_title=f"QM.{index + 1} TEST",
        text=f"Candidate passage {index}",
        word_count=3,
    )


if __name__ == "__main__":
    unittest.main()

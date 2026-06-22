from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re

from rag_chatbot.embedding_layer import IndexedChunk, chunk_to_embedding_text


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")
COMPACT_CODE_PATTERN = re.compile(r"\b([a-z]{2})(\d+[a-z]?)\b", re.IGNORECASE)
QUERY_STOP_WORDS = {
    "a",
    "an",
    "are",
    "can",
    "define",
    "describe",
    "explain",
    "for",
    "is",
    "of",
    "please",
    "the",
    "what",
}


@dataclass(frozen=True)
class KeywordSearchResult:
    """A chunk retrieved with a BM25 keyword score."""

    score: float
    chunk: IndexedChunk


class BM25Index:
    """Small in-memory BM25 index over saved chunk metadata."""

    def __init__(
        self,
        chunks: list[IndexedChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not chunks:
            raise ValueError("Cannot build a BM25 index with zero chunks")

        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.term_frequencies = [
            Counter(tokenize(chunk_to_embedding_text(chunk))) for chunk in chunks
        ]
        self.document_lengths = [
            sum(frequencies.values()) for frequencies in self.term_frequencies
        ]
        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths)
        )

        document_frequencies: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            document_frequencies.update(frequencies.keys())
        self.inverse_document_frequencies = {
            term: math.log(
                1
                + (
                    len(chunks) - document_frequency + 0.5
                )
                / (document_frequency + 0.5)
            )
            for term, document_frequency in document_frequencies.items()
        }

    def search(self, query: str, *, top_k: int = 10) -> list[KeywordSearchResult]:
        """Return the chunks with the highest BM25 scores."""
        query_terms = tokenize_query(query)
        if not query_terms:
            raise ValueError("query must contain searchable terms")

        scores = [
            self.score_document(query_terms, index)
            for index in range(len(self.chunks))
        ]
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )
        return [
            KeywordSearchResult(score=scores[index], chunk=self.chunks[index])
            for index in ranked_indices[:top_k]
            if scores[index] > 0
        ]

    def score_document(self, query_terms: list[str], index: int) -> float:
        """Calculate the BM25 score for one document."""
        frequencies = self.term_frequencies[index]
        document_length = self.document_lengths[index]
        chunk_text = self.chunks[index].text.lower().lstrip()
        score = 0.0

        for term in query_terms:
            term_frequency = frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            inverse_document_frequency = self.inverse_document_frequencies.get(
                term,
                0.0,
            )
            length_normalization = self.k1 * (
                1
                - self.b
                + self.b * document_length / self.average_document_length
            )
            score += inverse_document_frequency * (
                term_frequency * (self.k1 + 1)
            ) / (term_frequency + length_normalization)
            if "." in term and chunk_text.startswith(term):
                score += inverse_document_frequency
        return score


def tokenize(text: str) -> list[str]:
    """Tokenize prose while preserving codes such as QM.1 and SR.1a."""
    normalized = normalize_requirement_codes(text)
    return TOKEN_PATTERN.findall(normalized.lower())


def tokenize_query(text: str) -> list[str]:
    """Normalize exact codes and remove low-value question words."""
    return [
        token
        for token in tokenize(text)
        if token not in QUERY_STOP_WORDS
    ]


def normalize_requirement_codes(text: str) -> str:
    """Convert compact section codes such as QM1 and SR1a to dotted form."""
    return COMPACT_CODE_PATTERN.sub(r"\1.\2", text)

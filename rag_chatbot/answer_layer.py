from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Protocol, Sequence

from rag_chatbot.embedding_layer import IndexedChunk


DEFAULT_ANSWER_MODEL = "gemini-2.5-flash"

ANSWER_SYSTEM_PROMPT = """You answer questions about hospital accreditation.

Rules:
- Answer only from the supplied source passages.
- Summarize the evidence in clear, practical language.
- Do not use outside knowledge or invent requirements.
- Cite every factual paragraph using the supplied source labels, for example
  [Source 1: QM.1, pages 13-17].
- If the sources do not contain enough evidence, say that the indexed document
  does not provide enough information.
- Keep the answer concise unless the question asks for detail.
"""


@dataclass(frozen=True)
class GeneratedAnswer:
    """A grounded answer and the source chunks supplied to the LLM."""

    text: str
    sources: list[IndexedChunk]


class RankedChunk(Protocol):
    """Minimum result shape required by answer generation."""

    chunk: IndexedChunk


class AnswerGenerator(Protocol):
    """Interface for grounded answer generators."""

    def generate(self, prompt: str) -> str: ...


class GeminiAnswerGenerator:
    """Generate a grounded answer with Gemini."""

    def __init__(
        self,
        model_name: str = DEFAULT_ANSWER_MODEL,
        *,
        api_key: str | None = None,
    ) -> None:
        try:
            from dotenv import load_dotenv
            from google import genai
            from google.genai import types
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing answer-generation dependency. Install project "
                "dependencies with `python -m pip install -e .`."
            ) from exc

        load_dotenv()
        resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not resolved_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is required for answer generation."
            )

        self.model_name = model_name
        self.types = types
        self.client = genai.Client(api_key=resolved_api_key)

    def generate(self, prompt: str) -> str:
        """Call Gemini with the grounded-answer system prompt."""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self.types.GenerateContentConfig(
                system_instruction=ANSWER_SYSTEM_PROMPT,
                temperature=0.1,
            ),
        )
        answer = (response.text or "").strip()
        if not answer:
            raise RuntimeError("Answer model returned an empty response")
        return answer


def generate_grounded_answer(
    query: str,
    results: Sequence[RankedChunk],
    *,
    top_k: int = 3,
    model_name: str = DEFAULT_ANSWER_MODEL,
    generator: AnswerGenerator | None = None,
) -> GeneratedAnswer:
    """Send the top retrieved chunks to an LLM for grounded summarization."""
    if not query.strip():
        raise ValueError("query cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    sources = [result.chunk for result in results[:top_k]]
    if not sources:
        return GeneratedAnswer(
            text="The indexed document did not return evidence for this question.",
            sources=[],
        )

    prompt = build_answer_prompt(query, sources)
    active_generator = generator or GeminiAnswerGenerator(model_name)
    return GeneratedAnswer(
        text=active_generator.generate(prompt),
        sources=sources,
    )


def build_answer_prompt(query: str, sources: Sequence[IndexedChunk]) -> str:
    """Format the question and retrieved chunks as citation-ready context."""
    source_blocks = [
        format_source(number, chunk)
        for number, chunk in enumerate(sources, start=1)
    ]
    return "\n\n".join(
        [
            f"Question:\n{query.strip()}",
            "Retrieved sources:",
            *source_blocks,
            "Write the grounded answer now.",
        ]
    )


def format_source(number: int, chunk: IndexedChunk) -> str:
    """Format one retrieved chunk with an unambiguous citation label."""
    page_range = str(chunk.page_number)
    if chunk.end_page_number != chunk.page_number:
        page_range = f"{chunk.page_number}-{chunk.end_page_number}"

    label = (
        f"[Source {number}: {chunk.section_title}, "
        f"pages {page_range}]"
    )
    return "\n".join(
        [
            label,
            f"Chapter: {chunk.chapter_title}",
            f"Section: {chunk.section_title}",
            chunk.text,
        ]
    )

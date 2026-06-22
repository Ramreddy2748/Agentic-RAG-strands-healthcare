from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Protocol, Sequence

from pydantic import BaseModel, Field

from rag_chatbot.embedding_layer import IndexedChunk


DEFAULT_ANSWER_MODEL = "gemini-2.5-flash"

ANSWER_SYSTEM_PROMPT = """You format grounded hospital-accreditation evidence for clinicians.

Rules:
- Answer only from the supplied source passages.
- Use concise, clinically scannable language.
- Do not use outside knowledge or invent requirements.
- Cite only source numbers supplied in the prompt, such as 1, 2, or 3.
- Put source numbers only in each statement's citations array.
- Do not write citation numbers, brackets, or source labels inside statement text.
- Never invent section names, page numbers, or citation labels.
- Keep the summary to 2-3 sentences.
- Return no more than 6 key requirements and 5 clinical actions.
- Use limitations to state missing evidence or applicability concerns.
- Return JSON matching the required response schema.
"""


class CitedStatement(BaseModel):
    """One clinician-facing statement supported by numbered sources."""

    text: str = Field(min_length=1)
    citations: list[int] = Field(default_factory=list)


class ClinicalAnswer(BaseModel):
    """Structured, grounded answer designed for clinical scanning."""

    summary: CitedStatement
    key_requirements: list[CitedStatement] = Field(default_factory=list, max_length=6)
    clinical_actions: list[CitedStatement] = Field(default_factory=list, max_length=5)
    limitations: str | None = None


@dataclass(frozen=True)
class GeneratedAnswer:
    """A grounded answer and the source chunks supplied to the LLM."""

    content: ClinicalAnswer
    sources: list[IndexedChunk]


class RankedChunk(Protocol):
    """Minimum result shape required by answer generation."""

    chunk: IndexedChunk


class AnswerGenerator(Protocol):
    """Interface for grounded answer generators."""

    def generate(self, prompt: str) -> ClinicalAnswer: ...


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

    def generate(self, prompt: str) -> ClinicalAnswer:
        """Call Gemini with the grounded-answer system prompt."""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self.types.GenerateContentConfig(
                system_instruction=ANSWER_SYSTEM_PROMPT,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=ClinicalAnswer,
            ),
        )
        if response.parsed:
            return normalize_clinical_answer(
                ClinicalAnswer.model_validate(response.parsed)
            )

        answer_text = (response.text or "").strip()
        if not answer_text:
            raise RuntimeError("Answer model returned an empty response")
        return normalize_clinical_answer(
            ClinicalAnswer.model_validate_json(answer_text)
        )


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
            content=ClinicalAnswer(
                summary=CitedStatement(
                    text="The indexed document did not return evidence for this question.",
                    citations=[],
                ),
                limitations="No source passages were retrieved.",
            ),
            sources=[],
        )

    prompt = build_answer_prompt(query, sources)
    active_generator = generator or GeminiAnswerGenerator(model_name)
    content = active_generator.generate(prompt)
    validate_citations(content, source_count=len(sources))
    return GeneratedAnswer(
        content=content,
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
            (
                "Create a structured clinical response. Use only citation numbers "
                f"1 through {len(sources)}."
            ),
        ]
    )


def format_source(number: int, chunk: IndexedChunk) -> str:
    """Format one retrieved chunk with an unambiguous citation label."""
    page_range = str(chunk.page_number)
    if chunk.end_page_number != chunk.page_number:
        page_range = f"{chunk.page_number}-{chunk.end_page_number}"

    label = f"[Source {number}]"
    return "\n".join(
        [
            label,
            f"Chapter: {chunk.chapter_title}",
            f"Section: {chunk.section_title}",
            chunk.text,
        ]
    )


def validate_citations(answer: ClinicalAnswer, *, source_count: int) -> None:
    """Reject missing or invented source numbers in generated content."""
    statements = [
        answer.summary,
        *answer.key_requirements,
        *answer.clinical_actions,
    ]
    for statement in statements:
        if source_count > 0 and not statement.citations:
            raise ValueError("Every grounded statement must cite at least one source")
        invalid = [
            citation
            for citation in statement.citations
            if citation < 1 or citation > source_count
        ]
        if invalid:
            raise ValueError(f"Answer contains invalid source citations: {invalid}")


def normalize_clinical_answer(answer: ClinicalAnswer) -> ClinicalAnswer:
    """Remove redundant inline source numbers from structured statement text."""
    return ClinicalAnswer(
        summary=normalize_statement(answer.summary),
        key_requirements=[
            normalize_statement(statement)
            for statement in answer.key_requirements
        ],
        clinical_actions=[
            normalize_statement(statement)
            for statement in answer.clinical_actions
        ],
        limitations=answer.limitations,
    )


def normalize_statement(statement: CitedStatement) -> CitedStatement:
    """Keep citations in metadata instead of duplicating them in prose."""
    text = re.sub(
        r"\s*\((?:\d+\s*(?:,\s*\d+\s*)*)\)(?=[.,;:]|$)",
        "",
        statement.text,
    )
    return CitedStatement(
        text=text.strip(),
        citations=statement.citations,
    )

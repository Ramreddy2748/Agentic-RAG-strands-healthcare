from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
from threading import Lock
from time import monotonic, sleep
from typing import Callable, Iterable, Literal, Protocol

from pydantic import BaseModel, Field

from rag_chatbot.answer_layer import AnswerGenerator, ClinicalAnswer
from rag_chatbot.rag_service import RAGResponse


DEFAULT_EVALUATION_MODEL = "gemini-2.5-flash"

FAITHFULNESS_SYSTEM_PROMPT = """Evaluate whether each answer claim is supported by
its cited hospital-accreditation passages.

Rules:
- Judge only from the evidence supplied for that claim.
- Do not use outside medical or accreditation knowledge.
- Use "supported" when all material parts follow from the evidence.
- Use "partially_supported" when only part of the claim follows.
- Use "unsupported" when the evidence does not support the claim.
- Return exactly one judgment for every numbered claim.
- Keep reasons brief and evidence-focused.
"""


@dataclass(frozen=True)
class EvaluationCase:
    """One offline question with optional expected evidence."""

    case_id: str
    question: str
    expected_sections: tuple[str, ...] = ()
    expected_terms: tuple[str, ...] = ()
    search_mode: str = "auto"


@dataclass(frozen=True)
class CaseMetrics:
    """Deterministic quality and latency measurements for one response."""

    section_hit: bool | None
    section_recall: float | None
    first_relevant_rank: int | None
    reciprocal_rank: float | None
    evidence_term_recall: float | None
    answer_term_recall: float | None
    citations_valid: bool | None
    total_ms: float


@dataclass(frozen=True)
class ClaimFaithfulness:
    """Support judgment for one answer claim."""

    claim_number: int
    claim: str
    citations: tuple[int, ...]
    verdict: str
    reason: str


@dataclass(frozen=True)
class FaithfulnessEvaluation:
    """Faithfulness metrics produced with one judge call."""

    total_claims: int
    supported_claims: int
    partially_supported_claims: int
    unsupported_claims: int
    supported_claim_rate: float
    grounded_claim_rate: float
    judgments: tuple[ClaimFaithfulness, ...]


@dataclass(frozen=True)
class CaseEvaluation:
    """Evaluation output for one question."""

    case_id: str
    question: str
    search_mode: str
    retrieved_sections: tuple[str, ...]
    metrics: CaseMetrics
    faithfulness: FaithfulnessEvaluation | None = None
    error: str | None = None


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate metrics across successfully evaluated cases."""

    total_cases: int
    successful_cases: int
    failed_cases: int
    section_hit_rate: float | None
    mean_reciprocal_rank: float | None
    mean_section_recall: float | None
    mean_evidence_term_recall: float | None
    mean_answer_term_recall: float | None
    citation_validity_rate: float | None
    mean_supported_claim_rate: float | None
    mean_grounded_claim_rate: float | None
    mean_total_ms: float | None


@dataclass(frozen=True)
class EvaluationReport:
    """Serializable offline evaluation report."""

    summary: EvaluationSummary
    cases: tuple[CaseEvaluation, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ClaimJudgmentSchema(BaseModel):
    """Structured judgment returned by the evaluation model."""

    claim_number: int = Field(ge=1)
    verdict: Literal["supported", "partially_supported", "unsupported"]
    reason: str = Field(min_length=1)


class FaithfulnessResponseSchema(BaseModel):
    """One model response covering every claim in an answer."""

    judgments: list[ClaimJudgmentSchema]


class FaithfulnessJudge(Protocol):
    """Interface for one-call answer faithfulness judges."""

    def judge(self, prompt: str) -> FaithfulnessResponseSchema: ...


class GeminiFaithfulnessJudge:
    """Judge answer claims against their cited passages with Gemini."""

    def __init__(
        self,
        model_name: str = DEFAULT_EVALUATION_MODEL,
        *,
        api_key: str | None = None,
    ) -> None:
        try:
            from dotenv import load_dotenv
            from google import genai
            from google.genai import types
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing evaluation dependency. Install project dependencies "
                "with `python -m pip install -e .`."
            ) from exc

        load_dotenv()
        resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not resolved_api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for faithfulness evaluation.")

        self.model_name = model_name
        self.types = types
        self.client = genai.Client(api_key=resolved_api_key)

    def judge(self, prompt: str) -> FaithfulnessResponseSchema:
        """Make exactly one judge request without automatic application retries."""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self.types.GenerateContentConfig(
                system_instruction=FAITHFULNESS_SYSTEM_PROMPT,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=FaithfulnessResponseSchema,
            ),
        )
        if response.parsed:
            return FaithfulnessResponseSchema.model_validate(response.parsed)
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Faithfulness model returned an empty response")
        return FaithfulnessResponseSchema.model_validate_json(text)


class RequestRateLimiter:
    """Enforce a minimum interval between local API call starts."""

    def __init__(
        self,
        minimum_interval_seconds: float = 15.0,
        *,
        clock: Callable[[], float] = monotonic,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        if minimum_interval_seconds < 0:
            raise ValueError("minimum_interval_seconds cannot be negative")
        self.minimum_interval_seconds = minimum_interval_seconds
        self.clock = clock
        self.sleeper = sleeper
        self._last_request_started: float | None = None
        self._lock = Lock()

    def wait(self) -> None:
        """Wait as needed, then reserve the next API request slot."""
        with self._lock:
            now = self.clock()
            if self._last_request_started is not None:
                remaining = (
                    self.minimum_interval_seconds
                    - (now - self._last_request_started)
                )
                if remaining > 0:
                    self.sleeper(remaining)
                    now = self.clock()
            self._last_request_started = now


class RateLimitedAnswerGenerator:
    """Apply the shared request limiter immediately before answer generation."""

    def __init__(
        self,
        generator: AnswerGenerator,
        limiter: RequestRateLimiter,
    ) -> None:
        self.generator = generator
        self.limiter = limiter

    def generate(self, prompt: str) -> ClinicalAnswer:
        self.limiter.wait()
        return self.generator.generate(prompt)


class RateLimitedFaithfulnessJudge:
    """Apply the shared request limiter immediately before judging."""

    def __init__(
        self,
        judge: FaithfulnessJudge,
        limiter: RequestRateLimiter,
    ) -> None:
        self.judge_instance = judge
        self.limiter = limiter

    def judge(self, prompt: str) -> FaithfulnessResponseSchema:
        self.limiter.wait()
        return self.judge_instance.judge(prompt)


def load_evaluation_cases(path: str | Path) -> list[EvaluationCase]:
    """Load newline-delimited JSON evaluation cases."""
    cases: list[EvaluationCase] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            case = EvaluationCase(
                case_id=str(payload["case_id"]).strip(),
                question=str(payload["question"]).strip(),
                expected_sections=tuple(payload.get("expected_sections", [])),
                expected_terms=tuple(payload.get("expected_terms", [])),
                search_mode=str(payload.get("search_mode", "auto")),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid evaluation case on line {line_number}: {exc}"
            ) from exc
        validate_case(case, line_number=line_number)
        cases.append(case)

    if not cases:
        raise ValueError("Evaluation file contains no cases")
    return cases


def validate_case(case: EvaluationCase, *, line_number: int | None = None) -> None:
    """Validate fields that control an evaluation run."""
    location = f" on line {line_number}" if line_number is not None else ""
    if not case.case_id:
        raise ValueError(f"case_id cannot be empty{location}")
    if not case.question:
        raise ValueError(f"question cannot be empty{location}")
    if case.search_mode not in {"auto", "semantic", "keyword", "hybrid"}:
        raise ValueError(f"Invalid search_mode{location}: {case.search_mode}")


def evaluate_response(
    case: EvaluationCase,
    response: RAGResponse,
    *,
    faithfulness: FaithfulnessEvaluation | None = None,
) -> CaseEvaluation:
    """Score one RAG response against its expected evidence."""
    section_titles = tuple(result.chunk.section_title for result in response.results)
    section_ranks = find_expected_section_ranks(
        case.expected_sections,
        section_titles,
    )
    found_ranks = [rank for rank in section_ranks if rank is not None]
    section_recall = (
        len(found_ranks) / len(section_ranks)
        if section_ranks
        else None
    )
    first_rank = min(found_ranks) if found_ranks else None

    evidence_text = " ".join(result.chunk.text for result in response.results)
    answer_text = clinical_answer_text(response.answer)
    return CaseEvaluation(
        case_id=case.case_id,
        question=case.question,
        search_mode=response.search_mode,
        retrieved_sections=section_titles,
        metrics=CaseMetrics(
            section_hit=bool(found_ranks) if section_ranks else None,
            section_recall=round(section_recall, 4)
            if section_recall is not None
            else None,
            first_relevant_rank=first_rank,
            reciprocal_rank=round(1 / first_rank, 4) if first_rank else None,
            evidence_term_recall=term_recall(case.expected_terms, evidence_text),
            answer_term_recall=term_recall(case.expected_terms, answer_text)
            if response.answer is not None
            else None,
            citations_valid=citations_are_valid(
                response.answer,
                source_count=len(response.results),
            ),
            total_ms=response.timings.total_ms,
        ),
        faithfulness=faithfulness,
    )


def failed_evaluation(case: EvaluationCase, error: Exception) -> CaseEvaluation:
    """Record a failed case without aborting the complete batch."""
    return CaseEvaluation(
        case_id=case.case_id,
        question=case.question,
        search_mode=case.search_mode,
        retrieved_sections=(),
        metrics=CaseMetrics(
            section_hit=None,
            section_recall=None,
            first_relevant_rank=None,
            reciprocal_rank=None,
            evidence_term_recall=None,
            answer_term_recall=None,
            citations_valid=None,
            total_ms=0.0,
        ),
        error=f"{type(error).__name__}: {error}",
    )


def build_report(evaluations: Iterable[CaseEvaluation]) -> EvaluationReport:
    """Aggregate individual evaluations into one report."""
    cases = tuple(evaluations)
    successful = [case for case in cases if case.error is None]
    return EvaluationReport(
        summary=EvaluationSummary(
            total_cases=len(cases),
            successful_cases=len(successful),
            failed_cases=len(cases) - len(successful),
            section_hit_rate=mean_optional(
                case.metrics.section_hit for case in successful
            ),
            mean_reciprocal_rank=mean_optional(
                case.metrics.reciprocal_rank for case in successful
            ),
            mean_section_recall=mean_optional(
                case.metrics.section_recall for case in successful
            ),
            mean_evidence_term_recall=mean_optional(
                case.metrics.evidence_term_recall for case in successful
            ),
            mean_answer_term_recall=mean_optional(
                case.metrics.answer_term_recall for case in successful
            ),
            citation_validity_rate=mean_optional(
                case.metrics.citations_valid for case in successful
            ),
            mean_supported_claim_rate=mean_optional(
                case.faithfulness.supported_claim_rate
                if case.faithfulness is not None
                else None
                for case in successful
            ),
            mean_grounded_claim_rate=mean_optional(
                case.faithfulness.grounded_claim_rate
                if case.faithfulness is not None
                else None
                for case in successful
            ),
            mean_total_ms=mean_optional(
                case.metrics.total_ms for case in successful
            ),
        ),
        cases=cases,
    )


def find_expected_section_ranks(
    expected_sections: tuple[str, ...],
    retrieved_sections: tuple[str, ...],
) -> list[int | None]:
    """Return the first rank matching each expected section code or title."""
    normalized_retrieved = [normalize_text(value) for value in retrieved_sections]
    ranks: list[int | None] = []
    for expected in expected_sections:
        normalized_expected = normalize_text(expected)
        rank = next(
            (
                index
                for index, actual in enumerate(normalized_retrieved, start=1)
                if actual == normalized_expected
                or actual.startswith(f"{normalized_expected} ")
            ),
            None,
        )
        ranks.append(rank)
    return ranks


def term_recall(expected_terms: tuple[str, ...], text: str) -> float | None:
    """Measure the fraction of expected phrases found in normalized text."""
    if not expected_terms:
        return None
    normalized_text = normalize_text(text)
    matched = sum(
        normalize_text(term) in normalized_text
        for term in expected_terms
    )
    return round(matched / len(expected_terms), 4)


def citations_are_valid(
    answer: ClinicalAnswer | None,
    *,
    source_count: int,
) -> bool | None:
    """Check that every answer statement cites an available retrieved source."""
    if answer is None:
        return None
    statements = [
        answer.summary,
        *answer.key_requirements,
        *answer.clinical_actions,
    ]
    return all(
        statement.citations
        and all(1 <= citation <= source_count for citation in statement.citations)
        for statement in statements
    )


def clinical_answer_text(answer: ClinicalAnswer | None) -> str:
    """Flatten structured answer text for deterministic term checks."""
    if answer is None:
        return ""
    return " ".join(
        [
            answer.summary.text,
            *(statement.text for statement in answer.key_requirements),
            *(statement.text for statement in answer.clinical_actions),
            answer.limitations or "",
        ]
    )


def evaluate_faithfulness(
    response: RAGResponse,
    *,
    judge: FaithfulnessJudge,
) -> FaithfulnessEvaluation:
    """Judge all answer claims together using one model request."""
    if response.answer is None:
        raise ValueError("Faithfulness evaluation requires a generated answer")

    claims = answer_claims(response.answer)
    if not claims:
        raise ValueError("Generated answer contains no claims")

    prompt = build_faithfulness_prompt(response, claims)
    judged = judge.judge(prompt)
    judgments_by_number = {
        judgment.claim_number: judgment for judgment in judged.judgments
    }
    expected_numbers = set(range(1, len(claims) + 1))
    if set(judgments_by_number) != expected_numbers:
        raise ValueError(
            "Faithfulness judge must return exactly one judgment for every claim"
        )

    results = tuple(
        ClaimFaithfulness(
            claim_number=number,
            claim=claim.text,
            citations=tuple(claim.citations),
            verdict=judgments_by_number[number].verdict,
            reason=judgments_by_number[number].reason,
        )
        for number, claim in enumerate(claims, start=1)
    )
    supported = sum(result.verdict == "supported" for result in results)
    partial = sum(result.verdict == "partially_supported" for result in results)
    unsupported = sum(result.verdict == "unsupported" for result in results)
    total = len(results)
    return FaithfulnessEvaluation(
        total_claims=total,
        supported_claims=supported,
        partially_supported_claims=partial,
        unsupported_claims=unsupported,
        supported_claim_rate=round(supported / total, 4),
        grounded_claim_rate=round((supported + partial) / total, 4),
        judgments=results,
    )


def answer_claims(answer: ClinicalAnswer):
    """Return every structured answer statement in display order."""
    return [
        answer.summary,
        *answer.key_requirements,
        *answer.clinical_actions,
    ]


def build_faithfulness_prompt(response: RAGResponse, claims) -> str:
    """Include each claim with only the passages it cites."""
    blocks = [f"Question:\n{response.question}"]
    for number, claim in enumerate(claims, start=1):
        evidence_blocks = []
        for citation in claim.citations:
            if citation < 1 or citation > len(response.results):
                raise ValueError(f"Claim {number} has invalid citation {citation}")
            chunk = response.results[citation - 1].chunk
            evidence_blocks.append(
                "\n".join(
                    [
                        f"Source {citation}",
                        f"Section: {chunk.section_title}",
                        f"Pages: {chunk.page_number}-{chunk.end_page_number}",
                        chunk.text,
                    ]
                )
            )
        blocks.append(
            "\n\n".join(
                [
                    f"Claim {number}: {claim.text}",
                    "Cited evidence:",
                    *evidence_blocks,
                ]
            )
        )
    blocks.append(f"Return judgments for claims 1 through {len(claims)}.")
    return "\n\n".join(blocks)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def mean_optional(values: Iterable[float | bool | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 4)

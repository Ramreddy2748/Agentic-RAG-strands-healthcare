from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from collections import defaultdict, deque
from hashlib import sha256
import hmac
import os
import re
from threading import Lock
from time import monotonic
from typing import Callable, Iterable, Protocol
import unicodedata

from rag_chatbot.answer_layer import CitedStatement, ClinicalAnswer


API_KEYS_ENV = "RAG_API_KEYS"
SENSITIVE_FIELD_NAMES = {
    "api_key",
    "authorization",
    "credential",
    "credentials",
    "password",
    "question",
    "query",
    "secret",
    "text",
    "token",
}
SENSITIVE_VALUE_PATTERN = re.compile(
    r"(?i)(?:\bbearer\s+[0-9A-Za-z._~+/-]{8,}|"
    r"\bAIza[0-9A-Za-z_-]{20,}|"
    r"\bsk-[0-9A-Za-z_-]{16,}|"
    r"\b(?:api[_-]?key|token|password|secret)\s*[:=]\s*[^\s,;}]+)"
)
OUTPUT_LEAK_PATTERNS = (
    re.compile(
        r"\b(?:system prompt|developer message|hidden instructions?|"
        r"internal instructions?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:api key|access token|password|credential|environment variable)\b",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class PromptInjectionFinding:
    """One locally detected prompt-injection technique."""

    category: str
    description: str


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    """Non-secret identity used for authorization and rate limiting."""

    identifier: str
    auth_mode: str


@dataclass(frozen=True)
class EvidenceAssessment:
    """Decision describing whether retrieved evidence is strong enough."""

    sufficient: bool
    score: float
    threshold: float
    reason: str


@dataclass(frozen=True)
class OutputSafetyAssessment:
    """Decision describing whether generated output may be returned."""

    safe: bool
    reason: str | None = None


class RankedEvidence(Protocol):
    """Minimum result fields used by evidence confidence checks."""

    score: float


class SlidingWindowRateLimiter:
    """Small in-memory per-identity request limiter."""

    def __init__(
        self,
        requests: int,
        window_seconds: float,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if requests < 1:
            raise ValueError("requests must be at least 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.requests = requests
        self.window_seconds = window_seconds
        self.clock = clock
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, identity: str) -> tuple[bool, int]:
        """Consume one request or return the required retry delay."""
        now = self.clock()
        cutoff = now - self.window_seconds
        with self._lock:
            events = self._events[identity]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.requests:
                retry_after = max(1, int(self.window_seconds - (now - events[0])) + 1)
                return False, retry_after
            events.append(now)
            return True, 0


INJECTION_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        "Attempts to override trusted instructions.",
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass)\b.{0,60}"
            r"\b(?:previous|prior|above|system|developer|original|safety)\b"
            r".{0,30}\b(?:instruction|instructions|prompt|rules?|policy|policies)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "role_override",
        "Attempts to assign a new privileged model role.",
        re.compile(
            r"\b(?:act|behave|respond|pretend)\s+as\b.{0,50}"
            r"\b(?:system|developer|administrator|unrestricted|jailbroken)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "prompt_extraction",
        "Attempts to extract hidden prompts or instructions.",
        re.compile(
            r"\b(?:reveal|show|print|repeat|expose|provide|display)\b.{0,60}"
            r"\b(?:system prompt|developer message|hidden instructions?|"
            r"internal instructions?|initial prompt)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "secret_extraction",
        "Attempts to extract credentials or confidential configuration.",
        re.compile(
            r"\b(?:reveal|show|print|expose|provide|display|leak)\b.{0,60}"
            r"\b(?:api keys?|tokens?|passwords?|credentials?|secrets?|"
            r"environment variables?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "jailbreak",
        "Contains common jailbreak or safety-bypass language.",
        re.compile(
            r"\b(?:jailbreak|developer mode|do anything now|DAN mode|"
            r"without restrictions|disable safety|bypass safety)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_delimiter",
        "Contains forged model-role delimiters.",
        re.compile(
            r"(?:<\|(?:system|developer|assistant|user)\|>|"
            r"\[(?:system|developer|assistant)\]|"
            r"^(?:system|developer|assistant)\s*:)",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
)


def configured_api_keys(value: str | None = None) -> tuple[str, ...]:
    """Load non-empty API keys from a comma-separated environment value."""
    raw_value = os.getenv(API_KEYS_ENV, "") if value is None else value
    return tuple(
        key
        for item in raw_value.split(",")
        if (key := item.strip())
    )


def api_key_is_valid(
    provided_key: str | None,
    valid_keys: tuple[str, ...],
) -> bool:
    """Compare a presented API key without exposing timing differences."""
    if not provided_key or not valid_keys:
        return False
    return any(
        hmac.compare_digest(provided_key, expected_key)
        for expected_key in valid_keys
    )


def authenticate_principal(
    *,
    api_key: str | None,
    proxy_secret: str | None,
    proxy_user: str | None,
    auth_mode: str | None = None,
) -> AuthenticatedPrincipal | None:
    """Authenticate a local API key or a trusted upstream identity."""
    mode = (auth_mode or os.getenv("AUTH_MODE", "api_key")).strip().lower()
    if mode == "api_key":
        keys = configured_api_keys()
        if not api_key_is_valid(api_key, keys):
            return None
        digest = sha256((api_key or "").encode("utf-8")).hexdigest()[:16]
        return AuthenticatedPrincipal(
            identifier=f"api-key:{digest}",
            auth_mode=mode,
        )
    if mode == "trusted_proxy":
        expected_secret = os.getenv("TRUSTED_PROXY_SECRET", "")
        if (
            not expected_secret
            or not proxy_secret
            or not hmac.compare_digest(proxy_secret, expected_secret)
            or not proxy_user
            or not proxy_user.strip()
        ):
            return None
        digest = sha256(proxy_user.strip().encode("utf-8")).hexdigest()[:16]
        return AuthenticatedPrincipal(
            identifier=f"proxy-user:{digest}",
            auth_mode=mode,
        )
    raise ValueError(f"Unsupported AUTH_MODE: {mode}")


def detect_prompt_injection(question: str) -> list[PromptInjectionFinding]:
    """Detect explicit instruction override and secret-extraction techniques."""
    normalized = normalize_security_text(question)
    findings: list[PromptInjectionFinding] = []
    for category, description, pattern in INJECTION_PATTERNS:
        if pattern.search(normalized):
            findings.append(
                PromptInjectionFinding(
                    category=category,
                    description=description,
                )
            )
    return findings


def assess_retrieval_confidence(
    results: Iterable[object],
    *,
    search_mode: str,
    reranked: bool,
) -> EvidenceAssessment:
    """Apply mode-aware relevance thresholds to the best retrieved result."""
    result_list = list(results)
    if not result_list:
        return EvidenceAssessment(
            sufficient=False,
            score=0.0,
            threshold=0.0,
            reason="No evidence passages were retrieved.",
        )

    if reranked:
        score = max(float(getattr(result, "rerank_score", 0.0)) for result in result_list)
        threshold = env_float("MIN_RERANK_SCORE", 0.5)
        label = "reranker"
    elif search_mode == "semantic":
        score = max(float(getattr(result, "score", 0.0)) for result in result_list)
        threshold = env_float("MIN_SEMANTIC_SCORE", 0.35)
        label = "semantic"
    elif search_mode == "keyword":
        score = max(float(getattr(result, "score", 0.0)) for result in result_list)
        threshold = env_float("MIN_KEYWORD_SCORE", 0.01)
        label = "keyword"
    else:
        vector_score = max(
            float(getattr(result, "vector_score", 0.0)) for result in result_list
        )
        keyword_score = max(
            float(getattr(result, "keyword_score", 0.0)) for result in result_list
        )
        vector_threshold = env_float("MIN_SEMANTIC_SCORE", 0.35)
        keyword_threshold = env_float("MIN_KEYWORD_SCORE", 0.01)
        sufficient = (
            vector_score >= vector_threshold or keyword_score >= keyword_threshold
        )
        return EvidenceAssessment(
            sufficient=sufficient,
            score=max(vector_score, keyword_score),
            threshold=min(vector_threshold, keyword_threshold),
            reason=(
                "Hybrid evidence met a semantic or keyword confidence threshold."
                if sufficient
                else "Hybrid evidence did not meet semantic or keyword thresholds."
            ),
        )

    sufficient = score >= threshold
    return EvidenceAssessment(
        sufficient=sufficient,
        score=round(score, 4),
        threshold=threshold,
        reason=(
            f"Top {label} score met the configured threshold."
            if sufficient
            else f"Top {label} score was below the configured threshold."
        ),
    )


def insufficient_evidence_answer(reason: str) -> ClinicalAnswer:
    """Return a stable response without asking an LLM to fill evidence gaps."""
    return ClinicalAnswer(
        summary=CitedStatement(
            text=(
                "The indexed accreditation document did not provide sufficiently "
                "strong evidence to answer this question reliably."
            ),
            citations=[],
        ),
        limitations=reason,
    )


def assess_output_safety(answer: ClinicalAnswer | None) -> OutputSafetyAssessment:
    """Block prompt leakage, credential language, and configured secret values."""
    if answer is None:
        return OutputSafetyAssessment(safe=True)
    text = " ".join(
        [
            answer.summary.text,
            *(statement.text for statement in answer.key_requirements),
            *(statement.text for statement in answer.clinical_actions),
            answer.limitations or "",
        ]
    )
    for pattern in OUTPUT_LEAK_PATTERNS:
        if pattern.search(text):
            return OutputSafetyAssessment(
                safe=False,
                reason="Generated answer contained restricted internal information.",
            )
    for secret in configured_secret_values():
        if secret and secret in text:
            return OutputSafetyAssessment(
                safe=False,
                reason="Generated answer contained a configured secret value.",
            )
    return OutputSafetyAssessment(safe=True)


def safe_output_answer(reason: str) -> ClinicalAnswer:
    """Replace unsafe generated output with a non-sensitive response."""
    return ClinicalAnswer(
        summary=CitedStatement(
            text="The generated response was withheld by the output safety policy.",
            citations=[],
        ),
        limitations=reason,
    )


def configured_cors_origins(value: str | None = None) -> tuple[str, ...]:
    """Parse explicit browser origins; wildcard origins are not accepted."""
    raw_value = os.getenv("CORS_ALLOWED_ORIGINS", "") if value is None else value
    origins = tuple(
        origin
        for item in raw_value.split(",")
        if (origin := item.strip())
    )
    if "*" in origins:
        raise ValueError("CORS_ALLOWED_ORIGINS cannot contain '*'")
    return origins


def sanitize_log_data(value: object, *, field_name: str | None = None) -> object:
    """Recursively redact secrets and sensitive request content from logs."""
    if field_name and field_name.casefold() in SENSITIVE_FIELD_NAMES:
        return "[REDACTED]"
    if is_dataclass(value) and not isinstance(value, type):
        return sanitize_log_data(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): sanitize_log_data(item, field_name=str(key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [sanitize_log_data(item) for item in value]
    if isinstance(value, str):
        sanitized = value
        for secret in configured_secret_values():
            if secret:
                sanitized = sanitized.replace(secret, "[REDACTED]")
        return SENSITIVE_VALUE_PATTERN.sub("[REDACTED]", sanitized)
    return value


def configured_secret_values() -> tuple[str, ...]:
    """Return configured credentials that must never appear in output or logs."""
    values = [
        os.getenv("GOOGLE_API_KEY", ""),
        os.getenv("TRUSTED_PROXY_SECRET", ""),
        *configured_api_keys(),
    ]
    return tuple(value for value in values if len(value) >= 8)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def normalize_security_text(value: str) -> str:
    """Normalize Unicode and spacing before applying security patterns."""
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", normalized).strip()

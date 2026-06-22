from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Protocol


DEFAULT_ROUTER_MODEL = "gemini-2.5-flash"
VALID_SEARCH_MODES = {"semantic", "keyword", "hybrid"}

ROUTER_SYSTEM_PROMPT = """You route hospital accreditation questions to retrieval.

Choose exactly one search mode:
- semantic: Natural-language or conceptual questions where meaning matters more
  than exact wording, and the query does not include an exact code, citation,
  quoted phrase, or specialized identifier.
- keyword: Queries dominated by exact section codes, regulatory citations,
  abbreviations, quoted phrases, or identifiers, with little or no
  natural-language intent.
- hybrid: Queries that combine a natural-language question with any exact
  section code, citation, abbreviation, quoted phrase, or specialized
  identifier. Also use hybrid for broad or multi-part questions.

Examples:
- "What is quality management?" -> semantic
- "QM.1 SR.1a QAPI" -> keyword
- "What responsibilities does leadership have under QM.1?" -> hybrid
- "Explain QAPI requirements for hospital leadership" -> hybrid
- "\"management representative\"" -> keyword

Return JSON only:
{"mode": "semantic|keyword|hybrid", "reason": "short explanation"}

Do not answer the user's question. Only choose the retrieval mode.
"""


@dataclass(frozen=True)
class RoutingDecision:
    """The retrieval mode selected for a user query."""

    mode: str
    reason: str


class QueryRouter(Protocol):
    """Interface for query-routing implementations."""

    def route(self, query: str) -> RoutingDecision: ...


class GeminiQueryRouter:
    """Use a small Gemini call and a system prompt to select retrieval."""

    def __init__(
        self,
        model_name: str = DEFAULT_ROUTER_MODEL,
        *,
        api_key: str | None = None,
    ) -> None:
        try:
            from google import genai
            from google.genai import types
            from dotenv import load_dotenv
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing query-router dependency. Install project dependencies "
                "with `python -m pip install -e .`."
            ) from exc

        load_dotenv()
        resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not resolved_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is required for automatic search routing."
            )

        self.model_name = model_name
        self.types = types
        self.client = genai.Client(api_key=resolved_api_key)

    def route(self, query: str) -> RoutingDecision:
        """Ask Gemini to select semantic, keyword, or hybrid retrieval."""
        if not query.strip():
            raise ValueError("query cannot be empty")

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=query,
            config=self.types.GenerateContentConfig(
                system_instruction=ROUTER_SYSTEM_PROMPT,
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        return parse_routing_decision(response.text or "")


def route_query(
    query: str,
    *,
    router: QueryRouter | None = None,
    model_name: str = DEFAULT_ROUTER_MODEL,
    fallback_mode: str = "hybrid",
) -> RoutingDecision:
    """Route a query, falling back safely if the LLM router is unavailable."""
    if fallback_mode not in VALID_SEARCH_MODES:
        raise ValueError(f"Invalid fallback search mode: {fallback_mode}")

    try:
        active_router = router or GeminiQueryRouter(model_name)
        decision = active_router.route(query)
        if decision.mode not in VALID_SEARCH_MODES:
            raise ValueError(f"Router returned an invalid search mode: {decision.mode}")
        return decision
    except Exception as exc:
        return RoutingDecision(
            mode=fallback_mode,
            reason=f"Router unavailable; using {fallback_mode}. ({exc})",
        )


def parse_routing_decision(response_text: str) -> RoutingDecision:
    """Validate the JSON returned by the routing model."""
    payload = json.loads(response_text)
    mode = str(payload["mode"]).strip().lower()
    reason = str(payload.get("reason", "")).strip()
    if mode not in VALID_SEARCH_MODES:
        raise ValueError(f"Router returned an invalid search mode: {mode}")
    return RoutingDecision(mode=mode, reason=reason or "No reason provided.")

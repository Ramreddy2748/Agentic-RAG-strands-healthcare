"""Strands validation layer for grounded RAG answers.

This module is the public validation-layer entry point. The implementation
reuses the claim-grounding verifier in `verification_layer` so existing imports
and response fields remain backward compatible.
"""

from rag_chatbot.verification_layer import (
    AnswerVerifier,
    ClaimForVerification,
    ClaimVerification,
    StrandsVerificationAgent,
    VerificationMetadata,
    VerifiedAnswer,
    apply_verification as apply_validation,
    build_verification_prompt as build_validation_prompt,
    flatten_answer_claims,
    parse_agent_json,
    verify_clinical_answer as validate_clinical_answer,
    verification_is_enabled as validation_is_enabled,
)

__all__ = [
    "AnswerVerifier",
    "ClaimForVerification",
    "ClaimVerification",
    "StrandsVerificationAgent",
    "VerificationMetadata",
    "VerifiedAnswer",
    "apply_validation",
    "build_validation_prompt",
    "flatten_answer_claims",
    "parse_agent_json",
    "validate_clinical_answer",
    "validation_is_enabled",
]

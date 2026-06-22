from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from rag_chatbot.answer_layer import GeminiAnswerGenerator
from rag_chatbot.embedding_layer import DEFAULT_INDEX_DIR
from rag_chatbot.evaluation_layer import (
    DEFAULT_EVALUATION_MODEL,
    GeminiFaithfulnessJudge,
    RateLimitedAnswerGenerator,
    RateLimitedFaithfulnessJudge,
    RequestRateLimiter,
    build_report,
    evaluate_faithfulness,
    evaluate_response,
    failed_evaluation,
    load_evaluation_cases,
)
from rag_chatbot.rag_service import RAGService


DEFAULT_CASES_PATH = Path("evaluation/questions.example.jsonl")
DEFAULT_REPORT_PATH = Path(".rag_evaluation/report.json")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run an offline quality and latency evaluation of the RAG pipeline."
    )
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--output", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--generate-answers", action="store_true")
    parser.add_argument("--evaluate-faithfulness", action="store_true")
    parser.add_argument(
        "--evaluation-model",
        default=os.getenv("EVALUATION_MODEL", DEFAULT_EVALUATION_MODEL),
    )
    parser.add_argument("--api-interval-seconds", type=float, default=15.0)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    cases = load_evaluation_cases(args.cases)
    if args.evaluate_faithfulness and not args.generate_answers:
        parser.error("--evaluate-faithfulness requires --generate-answers")
    if args.generate_answers and any(
        case.search_mode == "auto" for case in cases
    ):
        parser.error(
            "Answer evaluation requires manual search_mode values to avoid "
            "additional router API calls."
        )

    limiter = RequestRateLimiter(args.api_interval_seconds)
    answer_generator = (
        RateLimitedAnswerGenerator(GeminiAnswerGenerator(), limiter)
        if args.generate_answers
        else None
    )
    faithfulness_judge = (
        RateLimitedFaithfulnessJudge(
            GeminiFaithfulnessJudge(args.evaluation_model),
            limiter,
        )
        if args.evaluate_faithfulness
        else None
    )
    service = RAGService.from_index_dir(
        args.index_dir,
        answer_generator=answer_generator,
    )
    evaluations = []

    for position, case in enumerate(cases, start=1):
        print(f"[{position}/{len(cases)}] {case.case_id}: {case.question}")
        try:
            response = service.ask(
                case.question,
                search_mode=case.search_mode,
                top_k=args.top_k,
                rerank=not args.no_rerank,
                generate_answer=args.generate_answers,
                answer_top_k=min(3, args.top_k),
                request_id=f"eval-{case.case_id}",
            )
            faithfulness = (
                evaluate_faithfulness(response, judge=faithfulness_judge)
                if faithfulness_judge is not None
                else None
            )
            evaluations.append(
                evaluate_response(
                    case,
                    response,
                    faithfulness=faithfulness,
                )
            )
        except Exception as exc:
            evaluations.append(failed_evaluation(case, exc))
            print(f"  failed: {type(exc).__name__}: {exc}")

    report = build_report(evaluations)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), indent=2),
        encoding="utf-8",
    )

    summary = report.summary
    print()
    print(f"Cases: {summary.successful_cases}/{summary.total_cases} successful")
    print(f"Section hit rate: {format_metric(summary.section_hit_rate)}")
    print(f"Mean reciprocal rank: {format_metric(summary.mean_reciprocal_rank)}")
    print(f"Mean section recall: {format_metric(summary.mean_section_recall)}")
    print(f"Mean evidence term recall: {format_metric(summary.mean_evidence_term_recall)}")
    print(f"Mean answer term recall: {format_metric(summary.mean_answer_term_recall)}")
    print(f"Citation validity: {format_metric(summary.citation_validity_rate)}")
    print(
        "Fully supported claim rate: "
        f"{format_metric(summary.mean_supported_claim_rate)}"
    )
    print(
        "Grounded claim rate: "
        f"{format_metric(summary.mean_grounded_claim_rate)}"
    )
    print(f"Mean latency: {format_metric(summary.mean_total_ms, suffix=' ms')}")
    print(f"Report: {output_path}")


def format_metric(value: float | None, *, suffix: str = "") -> str:
    if value is None:
        return "not measured"
    return f"{value:.4f}{suffix}"


if __name__ == "__main__":
    main()

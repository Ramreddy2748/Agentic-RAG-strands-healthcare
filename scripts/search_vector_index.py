from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_chatbot.answer_layer import DEFAULT_ANSWER_MODEL
from rag_chatbot.embedding_layer import DEFAULT_INDEX_DIR
from rag_chatbot.rag_service import RAGResponse, RAGService
from rag_chatbot.reranking_layer import DEFAULT_RERANKER_MODEL
from rag_chatbot.routing_layer import DEFAULT_ROUTER_MODEL


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the local hybrid index.")
    parser.add_argument("query")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--search-mode",
        choices=("auto", "hybrid", "semantic", "keyword"),
        default="auto",
    )
    parser.add_argument("--router-model", default=DEFAULT_ROUTER_MODEL)
    parser.add_argument(
        "--router-fallback",
        choices=("hybrid", "semantic", "keyword"),
        default="hybrid",
    )
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--keyword-weight", type=float, default=1.0)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--reranker-batch-size", type=int, default=2)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL)
    parser.add_argument("--show-results-only", action="store_true")
    args = parser.parse_args()

    service = RAGService.from_index_dir(
        args.index_dir,
        router_model=args.router_model,
        reranker_model=args.reranker_model,
        answer_model=args.answer_model,
    )
    response = service.ask(
        args.query,
        search_mode=args.search_mode,
        router_fallback=args.router_fallback,
        top_k=args.top_k,
        embedding_batch_size=args.batch_size,
        semantic_weight=args.semantic_weight,
        keyword_weight=args.keyword_weight,
        rerank=not args.no_rerank,
        reranker_batch_size=args.reranker_batch_size,
        reranker_max_length=args.reranker_max_length,
        generate_answer=not args.show_results_only,
        answer_top_k=3,
    )
    print_response(response, reranked=not args.no_rerank)


def print_response(response: RAGResponse, *, reranked: bool) -> None:
    """Render a structured service response for terminal debugging."""
    print(f"Request ID: {response.request_id}")
    print(
        f"Router selected: {response.search_mode} "
        f"({response.routing_reason})"
    )
    print(
        "Retrieval candidates: "
        f"semantic={response.stats.semantic_candidates}, "
        f"keyword={response.stats.keyword_candidates}, "
        f"combined={response.stats.fused_candidates}, "
        f"final={response.stats.final_results}"
    )
    print(
        "Timings (ms): "
        f"routing={response.timings.routing_ms}, "
        f"retrieval={response.timings.retrieval_ms}, "
        f"fusion={response.timings.fusion_ms}, "
        f"reranking={response.timings.reranking_ms}, "
        f"answer={response.timings.answer_generation_ms}, "
        f"total={response.timings.total_ms}"
    )

    for number, result in enumerate(response.results, start=1):
        chunk = result.chunk
        page_range = str(chunk.page_number)
        if chunk.end_page_number != chunk.page_number:
            page_range = f"{chunk.page_number}-{chunk.end_page_number}"

        print()
        if reranked:
            score_text = (
                f"rerank_score={result.rerank_score:.4f} "
                f"retrieval_score={result.hybrid_score:.4f} "
                f"vector_score={result.vector_score:.4f} "
                f"keyword_score={result.keyword_score:.4f}"
            )
        else:
            score_text = f"retrieval_score={result.score:.4f}"
            if hasattr(result, "vector_score"):
                score_text += f" vector_score={result.vector_score:.4f}"
            if hasattr(result, "keyword_score"):
                score_text += f" keyword_score={result.keyword_score:.4f}"
        print(f"[{number}] {score_text} pages={page_range}")
        print(f"Chapter: {chunk.chapter_title}")
        print(f"Section: {chunk.section_title}")
        print(chunk.text[:900])

    if response.answer is not None:
        print()
        print("Grounded answer:")
        print(response.answer.summary.text)
        if response.answer.key_requirements:
            print("\nKey requirements:")
            for item in response.answer.key_requirements:
                citations = ", ".join(f"[{number}]" for number in item.citations)
                print(f"- {item.text} {citations}")
        if response.answer.clinical_actions:
            print("\nClinical actions:")
            for item in response.answer.clinical_actions:
                citations = ", ".join(f"[{number}]" for number in item.citations)
                print(f"- {item.text} {citations}")
        if response.answer.limitations:
            print(f"\nLimitations: {response.answer.limitations}")


if __name__ == "__main__":
    main()

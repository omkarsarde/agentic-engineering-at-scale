"""Run Chapter 14's one end-to-end hybrid RAG experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from rag_pipeline import (
    BM25,
    DenseIndex,
    answer_or_abstain,
    assemble_context,
    average_precision_at_k,
    chunk_documents,
    citation_support,
    load_documents,
    load_jsonl,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieve,
    terms,
)


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "ch14" / "corpus.jsonl"
GOLD = ROOT / "data" / "ch14" / "gold.jsonl"


def unique_sources(hits) -> list[str]:
    """Collapse multiple chunks from one source while preserving rank."""
    return list(dict.fromkeys(hit.source_id for hit in hits))


def token_f1(answer: str, reference: str) -> float:
    """Transparent lexical answer-similarity proxy for the offline harness."""
    predicted = set(terms(answer)) - {"the", "a", "an", "and", "to", "of", "is", "for"}
    expected = set(terms(reference)) - {"the", "a", "an", "and", "to", "of", "is", "for"}
    overlap = len(predicted & expected)
    if not predicted or not expected or not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def metrics_for(rankings: list[list[str]], relevant_sets: list[set[str]], k: int = 5) -> dict:
    """Aggregate deterministic ranked-retrieval metrics."""
    return {
        "recall_at_5": mean(recall_at_k(ranked, relevant, k) for ranked, relevant in zip(rankings, relevant_sets)),
        "mrr": mean(reciprocal_rank(ranked, relevant) for ranked, relevant in zip(rankings, relevant_sets)),
        "map_at_5": mean(average_precision_at_k(ranked, relevant, k) for ranked, relevant in zip(rankings, relevant_sets)),
        "ndcg_at_5": mean(ndcg_at_k(ranked, {item: 1.0 for item in relevant}, k) for ranked, relevant in zip(rankings, relevant_sets)),
    }


def evaluate(size: int, overlap: int) -> dict:
    documents = load_documents(CORPUS)
    gold = load_jsonl(GOLD)
    chunks = chunk_documents(documents, size=size, overlap=overlap)
    lookup = {chunk.chunk_id: chunk for chunk in chunks}
    bm25, dense = BM25(chunks), DenseIndex(chunks)
    answerable = [item for item in gold if item["answerable"]]
    relevant_sets = [set(item["relevant_sources"]) for item in answerable]
    modes = {name: [] for name in ("bm25", "dense", "hybrid", "reranked")}
    candidate_recalls = []
    faithfulness, answer_similarity, context_precision = [], [], []
    citation_validity, citation_support_scores, abstention_correct = [], [], []
    rows = []

    for item in gold:
        query, history = item["query"], item.get("history", "")
        sparse_hits = bm25.search(query, 12)
        dense_hits = dense.search(query, 12)
        fused, final = retrieve(query, history, bm25, dense)
        answer = answer_or_abstain(query, final, lookup)
        context = assemble_context(final, lookup)
        relevant = set(item["relevant_sources"])

        if item["answerable"]:
            modes["bm25"].append(unique_sources(sparse_hits[:5]))
            modes["dense"].append(unique_sources(dense_hits[:5]))
            modes["hybrid"].append(unique_sources(fused[:5]))
            modes["reranked"].append(unique_sources(final))
            candidate_recalls.append(recall_at_k(unique_sources(fused), relevant, 12))
            returned = unique_sources(final)
            context_precision.append(len(set(returned) & relevant) / max(len(returned), 1))
            supported = citation_support(answer["answer"], lookup)
            faithfulness.append(supported)
            citation_support_scores.append(supported)
            citation_validity.append(float(all(citation in lookup for citation in answer["citations"])))
            answer_similarity.append(token_f1(answer["answer"], item["reference_answer"]))
            abstention_correct.append(float(not answer["abstained"]))
        else:
            abstention_correct.append(float(answer["abstained"]))

        rows.append(
            {
                "id": item["id"],
                "query": query,
                "answerable": item["answerable"],
                "retrieved_sources": unique_sources(final),
                "answer": answer["answer"],
                "context_tokens": len(context.split()),
            }
        )

    retrieval = {name: metrics_for(rankings, relevant_sets) for name, rankings in modes.items()}
    return {
        "chunk_size_words": size,
        "overlap_words": overlap,
        "documents_total": len(documents),
        "active_documents": sum(document.active for document in documents),
        "chunks": len(chunks),
        "retrieval": retrieval,
        "candidate_recall_at_12": mean(candidate_recalls),
        "generation": {
            "context_recall_at_5": retrieval["reranked"]["recall_at_5"],
            "context_precision_at_5": mean(context_precision),
            "faithfulness": mean(faithfulness),
            "answer_similarity": mean(answer_similarity),
            "citation_validity": mean(citation_validity),
            "citation_support": mean(citation_support_scores),
            "abstention_accuracy": mean(abstention_correct),
        },
        "rows": rows,
    }


def run() -> dict:
    """Sweep chunking and return the standard configuration plus all evidence."""
    sweep = [evaluate(size, max(4, size // 5)) for size in (18, 32, 48, 80)]
    standard = next(result for result in sweep if result["chunk_size_words"] == 48)
    return {
        "fixture": {
            "kind": "deterministic mechanism probe",
            "dense_encoder": "signed feature hashing with declared concept aliases",
            "reranker": "deterministic joint token/concept scorer",
            "token_unit": "whitespace words for chunking",
        },
        "standard": standard,
        "chunk_sweep": [
            {
                "chunk_size_words": result["chunk_size_words"],
                "chunks": result["chunks"],
                "recall_at_5": result["generation"]["context_recall_at_5"],
                "faithfulness": result["generation"]["faithfulness"],
                "answer_similarity": result["generation"]["answer_similarity"],
            }
            for result in sweep
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run()
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)

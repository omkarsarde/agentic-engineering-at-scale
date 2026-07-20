"""Focused deterministic tests for Chapter 14's hybrid RAG build."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
CODE = ROOT / "code" / "ch14"
sys.path.insert(0, str(CODE))
sys.modules.pop("chapter_build", None)

from chapter_build import evaluate, run  # noqa: E402
from rag_pipeline import (  # noqa: E402
    BM25,
    DenseIndex,
    Hit,
    answer_or_abstain,
    assemble_context,
    average_precision_at_k,
    chunk_documents,
    citation_support,
    load_documents,
    ndcg_at_k,
    query_variants,
    recall_at_k,
    reciprocal_rank,
    retrieve,
    rrf_merge,
)


CORPUS = ROOT / "data" / "ch14" / "corpus.jsonl"


def fixture(size: int = 48):
    documents = load_documents(CORPUS)
    chunks = chunk_documents(documents, size=size, overlap=max(4, size // 5))
    return chunks, BM25(chunks), DenseIndex(chunks)


def test_ranked_metrics_have_distinct_meanings() -> None:
    ranked, relevant = ["c1", "c2", "c3", "c4"], {"c1", "c3"}
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, relevant, 4) == 1.0
    assert reciprocal_rank(ranked, relevant) == 1.0
    assert average_precision_at_k(ranked, relevant, 4) == pytest.approx((1 + 2 / 3) / 2)
    assert ndcg_at_k(ranked, {"c1": 2.0, "c3": 1.0}, 4) < 1.0


def test_bm25_surfaces_the_rare_error_code() -> None:
    _, bm25, _ = fixture()
    hits = bm25.search("E1492", 3)
    assert hits[0].source_id == "error-e1492-v5"
    assert bm25.idf("e1492") > bm25.idf("the")


def test_dense_retrieval_bridges_declared_paraphrases() -> None:
    _, _, dense = fixture()
    sources = {hit.source_id for hit in dense.search("stop my yearly membership from renewing", 5)}
    assert "cancel-annual-v4" in sources


def test_hybrid_and_rerank_improve_without_changing_candidate_set() -> None:
    chunks, bm25, dense = fixture()
    fused, final = retrieve("Can I get my money back six weeks after buying in the US?", "", bm25, dense)
    assert final[0].source_id == "refund-us-v3"
    assert {hit.chunk_id for hit in final}.issubset({hit.chunk_id for hit in fused})
    report = evaluate(48, 9)
    assert report["retrieval"]["hybrid"]["recall_at_5"] > report["retrieval"]["bm25"]["recall_at_5"]
    assert report["retrieval"]["reranked"]["mrr"] > report["retrieval"]["hybrid"]["mrr"]
    assert report["candidate_recall_at_12"] == 1.0


def test_retired_policy_is_not_indexed_and_chunk_ids_are_stable() -> None:
    documents = load_documents(CORPUS)
    first = chunk_documents(documents, 32, 6)
    second = chunk_documents(documents, 32, 6)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert all(chunk.source_id != "refund-us-v2" for chunk in first)


def test_rrf_depends_on_rank_not_incomparable_raw_scores() -> None:
    chunks, _, _ = fixture()
    lookup = {chunk.chunk_id: chunk for chunk in chunks}
    one, two = chunks[:2]
    a = [Hit(one.chunk_id, one.source_id, 1_000_000), Hit(two.chunk_id, two.source_id, -9)]
    b = [Hit(two.chunk_id, two.source_id, 0.01), Hit(one.chunk_id, one.source_id, 0.001)]
    merged = rrf_merge([a, b], lookup)
    assert merged[0].score == pytest.approx(merged[1].score)


def test_query_condensation_restores_the_missing_subject() -> None:
    variants = query_variants("What about annual plans?", "How can I cancel before renewal?")
    assert variants[-1] == "cancel annual subscription before renewal"


def test_context_is_budgeted_at_source_boundaries_with_provenance() -> None:
    chunks, bm25, dense = fixture()
    _, hits = retrieve("What does E1492 mean?", "", bm25, dense)
    lookup = {chunk.chunk_id: chunk for chunk in chunks}
    context = assemble_context(hits, lookup, budget=70)
    assert 'trust="retrieved-data"' in context
    assert context.count("<source ") == context.count("</source>")
    assert len(context.split()) <= 70


def test_citation_syntax_does_not_prove_support_and_unanswerables_abstain() -> None:
    chunks, bm25, dense = fixture()
    lookup = {chunk.chunk_id: chunk for chunk in chunks}
    _, refund_hits = retrieve("refund in the US", "", bm25, dense)
    real = answer_or_abstain("refund in the US", refund_hits, lookup)
    assert citation_support(real["answer"], lookup) == 1.0
    fake = f"Refunds are instant and unlimited. [{refund_hits[0].chunk_id}]"
    assert citation_support(fake, lookup) == 0.0
    _, unknown_hits = retrieve("When do gift cards expire?", "", bm25, dense)
    assert answer_or_abstain("When do gift cards expire?", unknown_hits, lookup)["abstained"]


def test_integrated_build_is_deterministic_and_reports_all_stages() -> None:
    first, second = run(), run()
    assert first == second
    assert len(first["chunk_sweep"]) == 4
    standard = first["standard"]
    assert standard["generation"]["citation_support"] == 1.0
    assert standard["generation"]["abstention_accuracy"] == 1.0
    assert standard["retrieval"]["reranked"]["ndcg_at_5"] >= standard["retrieval"]["hybrid"]["ndcg_at_5"]

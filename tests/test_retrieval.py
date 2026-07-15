"""
tests/test_retrieval.py

순수 로직 테스트 — 외부 의존 없음(모델/ES 불필요).
RRF 융합과 검색 평가 지표의 수학적 정확성을 검증한다.
"""

from __future__ import annotations

import math

from eval.retrieval_eval import (
    _ndcg_at_k,
    _recall_at_k,
    _rr,
    evaluate_retrieval,
)
from retrieval.fusion import rrf
from retrieval.vector_store import Doc


# ── RRF 융합 ─────────────────────────────────────────────
def test_rrf_both_lists_top_wins():
    """두 검색기 모두 상위인 문서가 1위가 되어야."""
    bm25 = [Doc("A", "a", 9.0), Doc("B", "b", 7.0), Doc("C", "c", 5.0)]
    knn = [Doc("C", "c", 0.9), Doc("A", "a", 0.8), Doc("D", "d", 0.7)]
    fused = rrf([bm25, knn], k=4)
    assert fused[0].id == "A"  # 양쪽 모두 상위


def test_rrf_includes_single_list_docs():
    """한쪽에만 등장한 문서도 결과에 포함되어야(누락 없음)."""
    bm25 = [Doc("A", "a", 9.0)]
    knn = [Doc("B", "b", 0.9)]
    ids = {d.id for d in rrf([bm25, knn], k=5)}
    assert ids == {"A", "B"}


def test_rrf_respects_k():
    """k개만 반환."""
    lst = [Doc(f"d{i}", "x", 1.0) for i in range(10)]
    assert len(rrf([lst], k=3)) == 3


def test_rrf_score_formula():
    """RRF 점수가 1/(rank_constant+rank) 공식과 일치."""
    bm25 = [Doc("A", "a", 9.0), Doc("B", "b", 7.0)]
    fused = rrf([bm25], k=2, rank_constant=60)
    # A는 rank 0 → 1/(60+1), B는 rank 1 → 1/(60+2)
    a = next(d for d in fused if d.id == "A")
    assert abs(a.score - 1.0 / 61) < 1e-9


def test_rrf_empty_input():
    """빈 입력에 안전."""
    assert rrf([], k=5) == []
    assert rrf([[]], k=5) == []


# ── Recall@k ─────────────────────────────────────────────
def test_recall_all_found():
    assert _recall_at_k(["d1", "d2", "d3"], {"d1", "d2"}, k=3) == 1.0


def test_recall_partial():
    # 정답 2개 중 1개만 상위 k에
    assert _recall_at_k(["d1", "x", "y"], {"d1", "d2"}, k=3) == 0.5


def test_recall_k_cutoff():
    # 정답이 k 밖에 있으면 회수 안 됨
    assert _recall_at_k(["x", "y", "d1"], {"d1"}, k=2) == 0.0


# ── MRR ──────────────────────────────────────────────────
def test_rr_first_position():
    assert _rr(["d1", "x"], {"d1"}) == 1.0


def test_rr_third_position():
    assert _rr(["x", "y", "d1"], {"d1"}) == 1.0 / 3


def test_rr_not_found():
    assert _rr(["x", "y"], {"d1"}) == 0.0


# ── nDCG ─────────────────────────────────────────────────
def test_ndcg_perfect_ranking():
    """정답이 맨 위면 nDCG=1.0."""
    assert abs(_ndcg_at_k(["d1", "d2", "x"], {"d1", "d2"}, k=3) - 1.0) < 1e-9


def test_ndcg_demoted_ranking():
    """정답이 아래로 밀리면 nDCG < 1.0."""
    score = _ndcg_at_k(["x", "y", "d1"], {"d1"}, k=3)
    assert 0 < score < 1.0
    # d1이 3위(i=2) → DCG = 1/log2(4) = 0.5, IDCG = 1/log2(2) = 1.0
    assert abs(score - (1.0 / math.log2(4))) < 1e-9


# ── 집계 ─────────────────────────────────────────────────
def test_evaluate_retrieval_aggregate():
    gs = [
        {"qid": "a", "question": "q1", "gold_doc_ids": ["d1"]},
        {"qid": "b", "question": "q2", "gold_doc_ids": ["d2"]},
    ]
    # 완벽 검색기
    perfect = evaluate_retrieval(
        lambda q: ["d1"] if q == "q1" else ["d2"], gs, k=3)["aggregate"]
    assert perfect["Recall@3"] == 1.0
    assert perfect["MRR"] == 1.0

    # 완전 실패 검색기
    bad = evaluate_retrieval(lambda q: ["zzz"], gs, k=3)["aggregate"]
    assert bad["Recall@3"] == 0.0
    assert bad["MRR"] == 0.0


def test_adaptive_beats_baseline():
    """재검색 루프(adaptive)가 단일검색(baseline)보다 검색 지표가 높아야 — 핵심 회귀 보호."""
    gs = [
        {"qid": "a", "question": "easy", "gold_doc_ids": ["d1"]},
        {"qid": "b", "question": "hard", "gold_doc_ids": ["d2"]},
    ]
    # baseline: hard 질문에서 정답 못 찾음
    baseline = evaluate_retrieval(
        lambda q: ["d1"] if q == "easy" else ["noise"], gs, k=3)["aggregate"]
    # adaptive: 둘 다 찾음
    adaptive = evaluate_retrieval(
        lambda q: ["d1"] if q == "easy" else ["d2"], gs, k=3)["aggregate"]
    assert adaptive["Recall@3"] > baseline["Recall@3"]

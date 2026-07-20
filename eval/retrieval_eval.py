"""
gemma-rag-harness / eval/retrieval_eval.py

검색 계층 단독 평가. 생성(LLM)과 분리해 '검색이 정답 문서를 가져오는가'만 본다.
재검색 루프 튜닝의 효과를 숫자로 증명하는 1차 지표.

지표:
  - Recall@k : 상위 k개 안에 정답 문서가 포함된 비율 (정답 doc 중 몇 %를 회수했나)
  - MRR      : 첫 정답 문서의 순위 역수 평균 (정답이 얼마나 위에 오나)
  - nDCG@k   : 순위 가중 정답 회수 (여러 정답의 순서까지 반영)
  - Hit@k    : 정답을 하나라도 맞춘 질의 비율

사용:
  from eval.retrieval_eval import evaluate_retrieval
  metrics = evaluate_retrieval(search_fn, goldenset, k=5)
  # search_fn(question) -> list[doc_id]  (순위 내림차순)
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path


def load_goldenset(path: str | Path) -> list[dict]:
    """JSONL 골든셋 로드. out_of_scope(정답 문서 없음)는 검색 평가에서 제외."""
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            if row.get("gold_doc_ids"):     # 검색 평가 대상만
                items.append(row)
    return items


def _recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(retrieved[:k])
    return len(top & gold) / len(gold)


def _rr(retrieved: list[str], gold: set[str]) -> float:
    """Reciprocal Rank: 첫 정답 문서의 1/순위."""
    for rank, did in enumerate(retrieved, start=1):
        if did in gold:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """이진 관련도 가정의 nDCG@k."""
    dcg = 0.0
    for i, did in enumerate(retrieved[:k]):
        if did in gold:
            dcg += 1.0 / math.log2(i + 2)       # i는 0부터 → 분모 log2(rank+1)
    # 이상적 DCG: 정답들이 맨 위에 몰린 경우
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    search_fn: Callable[[str], list[str]],
    goldenset: list[dict],
    k: int = 5,
) -> dict:
    """
    search_fn: 질문 → 검색된 doc_id 리스트(순위순).
    반환: 집계 지표 + 질의별 상세.
    """
    per_query = []
    for row in goldenset:
        gold = set(row["gold_doc_ids"])
        retrieved = search_fn(row["question"])
        per_query.append({
            "qid": row["qid"],
            "recall@k": _recall_at_k(retrieved, gold, k),
            "rr": _rr(retrieved, gold),
            "ndcg@k": _ndcg_at_k(retrieved, gold, k),
            "hit": 1.0 if set(retrieved[:k]) & gold else 0.0,
        })

    n = len(per_query) or 1
    agg = {
        f"Recall@{k}": round(sum(q["recall@k"] for q in per_query) / n, 4),
        "MRR": round(sum(q["rr"] for q in per_query) / n, 4),
        f"nDCG@{k}": round(sum(q["ndcg@k"] for q in per_query) / n, 4),
        f"Hit@{k}": round(sum(q["hit"] for q in per_query) / n, 4),
        "n_queries": len(per_query),
    }
    return {"aggregate": agg, "per_query": per_query}


if __name__ == "__main__":
    # 데모: 완벽 검색기 vs 무작위 검색기 비교
    gs = [
        {"qid": "a", "question": "q1", "gold_doc_ids": ["d1", "d2"]},
        {"qid": "b", "question": "q2", "gold_doc_ids": ["d3"]},
    ]
    def perfect(q: str) -> list[str]:
        return ["d1", "d2", "d3"] if q == "q1" else ["d3", "d9"]

    def bad(q: str) -> list[str]:
        return ["d7", "d8", "d9"]

    print("완벽 검색기:", evaluate_retrieval(perfect, gs, k=3)["aggregate"])
    print("무관 검색기:", evaluate_retrieval(bad, gs, k=3)["aggregate"])

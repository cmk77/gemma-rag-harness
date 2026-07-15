"""
gemma-rag-harness / retrieval/fusion.py

Reciprocal Rank Fusion (RRF).
여러 검색 결과 리스트를 순위 기반으로 융합한다.
점수 스케일이 다른 BM25와 kNN을 보정 없이 합칠 수 있는 게 핵심 장점.

  RRF_score(d) = Σ_lists  1 / (rank_constant + rank_in_list(d))

rank_constant(기본 60)는 상위 순위의 영향력을 조절하는 상수.
값이 클수록 순위 차이의 영향이 완만해진다(Cormack et al. 2009 권장값).
"""

from __future__ import annotations

from retrieval.vector_store import Doc


def rrf(result_lists: list[list[Doc]], k: int = 5, rank_constant: int = 60) -> list[Doc]:
    """
    result_lists: 각 검색기(BM25, kNN, ...)가 반환한 Doc 리스트들.
                  각 리스트는 이미 자기 기준 점수 내림차순이라고 가정.
    반환: RRF 점수 상위 k개 Doc (score 필드를 RRF 점수로 교체).
    """
    scores: dict[str, float] = {}
    docs_by_id: dict[str, Doc] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):          # rank: 0부터
            scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (rank_constant + rank + 1)
            docs_by_id.setdefault(doc.id, doc)        # 첫 등장 Doc의 메타 유지

    ranked_ids = sorted(scores, key=lambda i: scores[i], reverse=True)[:k]
    return [
        Doc(id=i, text=docs_by_id[i].text, score=scores[i], source=docs_by_id[i].source)
        for i in ranked_ids
    ]

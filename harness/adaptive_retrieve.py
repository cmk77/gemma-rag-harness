"""
gemma-rag-harness / harness/adaptive_retrieve.py

재검색 루프의 핵심. VERIFY가 "insufficient"를 내면 graph가 이 노드로 재진입하는데,
같은 입력으로 다시 검색하면 의미가 없으므로 retry 단계마다 검색 '입력'을 바꾼다.

전략 에스컬레이션 (검색 입력을 점점 더 크게 변형):
  retry 0 : 원 질의 그대로 + 기본 k
  retry 1 : rewrite (검색 친화적 재작성) + k 증가
  retry 2 : HyDE (가상 답변 임베딩) + decompose (하위 질의 병합) + k 더 증가

추가로:
  - 이전 retry에서 모은 문서를 버리지 않고 누적(accumulate)
  - id 기준 중복 제거 → 새 전략이 가져온 '새 문서'만 실질적으로 더해짐
  - 누적 문서가 일정 수를 넘으면 더 키우지 않음(컨텍스트 예산 보호)

graph.py는 retrieve 노드를 이 모듈의 adaptive_retrieve로 교체한다.
"""

from __future__ import annotations

from harness import query_expansion as qe
from retrieval.es_store import ESStore
from retrieval.indexer import embed, embed_query
from retrieval.vector_store import Doc

_STORE = ESStore()

BASE_K = 10                # 첫 검색 문서 수. 중의성 있는 질의(예: "인증"이
                           # certification/authentication 양쪽 의미)에서 정답
                           # 문서가 top-k에 안 들어오는 것을 막기 위해 넉넉히.
K_STEP = 3                 # retry마다 k 증가폭
MAX_CONTEXT_DOCS = 15      # 누적 문서 상한(컨텍스트 예산 보호). BASE_K 증가에 맞춰 상향.


def _dedup(docs: list[Doc]) -> list[Doc]:
    """id 기준 중복 제거, 첫 등장(보통 더 높은 RRF 점수) 우선."""
    seen: set[str] = set()
    out: list[Doc] = []
    for d in docs:
        if d.id not in seen:
            seen.add(d.id)
            out.append(d)
    return out


def _search_original(query: str, k: int) -> list[Doc]:
    return _STORE.hybrid_search(query, embed_query(query), k=k)


def _search_rewritten(query: str, k: int, diagnosis: str = "") -> list[Doc]:
    # 재작성 질의는 키워드(BM25)와 의미(kNN) 양쪽에 더 잘 맞는다.
    # VERIFY 진단이 있으면 '빠진 정보'를 겨냥해 재작성(없으면 일반 재작성).
    rq = qe.guided_rewrite(query, diagnosis)
    return _STORE.hybrid_search(rq, embed_query(rq), k=k)


def _search_hyde(query: str, k: int) -> list[Doc]:
    # 가상 답변을 임베딩해 dense 검색을 정답 문서 쪽으로 끌어당김.
    # BM25 측은 여전히 원 질의 키워드를 쓰는 게 안전.
    hypo = qe.hyde(query)
    return _STORE.hybrid_search(query, embed([hypo])[0], k=k)


def _search_decomposed(query: str, k: int) -> list[Doc]:
    # 하위 질의별로 검색해 합침(multi-hop 근거 수집).
    merged: list[Doc] = []
    for sub in qe.decompose(query):
        merged += _STORE.hybrid_search(sub, embed_query(sub), k=max(2, k // 2))
    return merged


def adaptive_retrieve(state: dict) -> dict:
    """
    graph.py의 AgentState를 받아 docs를 채워 반환.
    state 사용 키: query, retry_count, diagnosis(VERIFY가 남긴 부족점), _doc_pool(누적 풀)

    루프 가드: 이번 재검색이 '새 문서'를 하나도 못 가져오면 no_new_docs=True를
    반환한다. graph의 verify_decision이 이 신호를 보고 무의미한 재시도를 끊는다.
    """
    query: str = state["query"]
    retry: int = state.get("retry_count", 0)
    diagnosis: str = state.get("diagnosis", "")
    k = BASE_K + retry * K_STEP

    # 단계별 전략 선택 (retry가 오를수록 검색 입력을 더 크게 변형)
    if retry == 0:
        fresh = _search_original(query, k)
    elif retry == 1:
        fresh = _search_rewritten(query, k, diagnosis)   # 진단 겨냥 재작성
    else:  # retry >= 2: 가장 공격적 — HyDE + 분해를 함께
        fresh = _search_hyde(query, k) + _search_decomposed(query, k)

    prev_pool: list[Doc] = state.get("_doc_pool", [])
    prev_ids = {d.id for d in prev_pool}

    # 누적 후 중복 제거
    pool = _dedup(prev_pool + fresh)

    # 이번 라운드가 실제로 새 문서를 더했는지 (루프 가드 신호)
    new_ids = {d.id for d in pool} - prev_ids
    no_new_docs = len(new_ids) == 0 and retry > 0

    # RRF 점수 내림차순 정렬 후 컨텍스트 상한 적용
    pool.sort(key=lambda d: d.score, reverse=True)
    pool = pool[:MAX_CONTEXT_DOCS]

    return {
        "_doc_pool": pool,                       # 다음 재진입을 위해 보존
        "docs": [d.text for d in pool],          # generate가 쓰는 형태
        "retry_count": retry,                    # verify가 증가시킴
        "no_new_docs": no_new_docs,              # graph가 루프 가드에 사용
    }

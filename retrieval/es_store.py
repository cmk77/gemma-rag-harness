"""
gemma-rag-harness / retrieval/es_store.py

Elasticsearch 8.x 기반 VectorStore 구현.
- 색인: text(BM25용) + dense_vector(kNN용) 필드를 한 문서에 동시 저장
- 검색: BM25 결과와 kNN 결과를 각각 받아 애플리케이션 레벨 RRF로 융합
        (ES 8.x의 네이티브 rank: rrf 도 주석으로 병기 — 라이선스/버전에 따라 선택)

의존: pip install elasticsearch>=8.11
환경변수: ES_URL (기본 http://localhost:9200), ES_INDEX (기본 gemma_rag)
"""

from __future__ import annotations

import os

from elasticsearch import Elasticsearch, helpers

from retrieval.fusion import rrf
from retrieval.vector_store import Doc

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
INDEX = os.getenv("ES_INDEX", "gemma_rag")

# bge-m3 임베딩 차원. 임베딩 모델 교체 시 이 값도 함께 변경.
EMBED_DIM = 1024


# ─────────────────────────────────────────────────────────────
# 인덱스 매핑
#   - text: nori 한국어 형태소 분석기(한/영 혼용 코퍼스 대응)
#   - vector: dense_vector, cosine 유사도, kNN 활성화
# ─────────────────────────────────────────────────────────────
INDEX_MAPPING = {
    "settings": {
        "analysis": {
            "analyzer": {
                "kr": {"type": "custom", "tokenizer": "nori_tokenizer"}
            }
        }
    },
    "mappings": {
        "properties": {
            "text":   {"type": "text", "analyzer": "kr"},
            "source": {"type": "keyword"},
            "vector": {
                "type": "dense_vector",
                "dims": EMBED_DIM,
                "index": True,
                "similarity": "cosine",
            },
        }
    },
}


class ESStore:
    """VectorStore 구현 (Protocol 충족)."""

    def __init__(self, url: str = ES_URL, index: str = INDEX) -> None:
        self.es = Elasticsearch(url)
        self.index_name = index

    def ensure_index(self) -> None:
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name, body=INDEX_MAPPING)

    # ── 색인 ──────────────────────────────────────────────────
    def index(self, docs: list[Doc], vectors: list[list[float]] | None = None) -> None:
        """
        docs와 (선택) vectors를 받아 bulk 색인.
        indexer.py가 임베딩을 계산해 vectors로 넘긴다.
        """
        self.ensure_index()
        actions = []
        for i, d in enumerate(docs):
            body = {"text": d.text, "source": d.source}
            if vectors is not None:
                body["vector"] = vectors[i]
            actions.append({"_index": self.index_name, "_id": d.id, "_source": body})
        helpers.bulk(self.es, actions)
        self.es.indices.refresh(index=self.index_name)

    # ── 검색 ──────────────────────────────────────────────────
    def _bm25(self, query: str, k: int) -> list[Doc]:
        res = self.es.search(
            index=self.index_name,
            query={"match": {"text": query}},
            size=k,
        )
        return self._to_docs(res)

    def _knn(self, query_vector: list[float], k: int) -> list[Doc]:
        res = self.es.search(
            index=self.index_name,
            knn={
                "field": "vector",
                "query_vector": query_vector,
                "k": k,
                "num_candidates": max(50, k * 10),
            },
            size=k,
        )
        return self._to_docs(res)

    def hybrid_search(self, query: str, query_vector: list[float], k: int = 5) -> list[Doc]:
        """
        BM25 + kNN을 각각 넉넉히(k*2) 가져와 RRF로 융합 후 상위 k 반환.

        ── ES 네이티브 RRF 대안 (ES 8.8+, 적정 라이선스 필요) ──
        res = self.es.search(index=self.index_name, retriever={
            "rrf": {
                "retrievers": [
                    {"standard": {"query": {"match": {"text": query}}}},
                    {"knn": {"field": "vector", "query_vector": query_vector,
                             "k": k, "num_candidates": k*10}},
                ],
                "rank_window_size": k*2, "rank_constant": 60,
            }
        }, size=k)
        return self._to_docs(res)
        """
        bm25 = self._bm25(query, k * 2)
        knn = self._knn(query_vector, k * 2)
        fused = rrf([bm25, knn], k=k)
        return fused

    @staticmethod
    def _to_docs(res) -> list[Doc]:
        out = []
        for hit in res["hits"]["hits"]:
            src = hit["_source"]
            out.append(Doc(
                id=hit["_id"],
                text=src.get("text", ""),
                score=float(hit.get("_score") or 0.0),
                source=src.get("source", ""),
            ))
        return out

"""Elasticsearch 색인/검색.

Progressive Disclosure를 색인 구조로 구현한다:
  1단계 meta_search : title/description/tags/type만 대상 (본문 미노출)
  2단계 hybrid_search: 본문 BM25 + kNN을 수동 RRF로 융합
필드 컨벤션은 하네스와 동일 — 본문 필드명은 반드시 `text`
(과거 진단 실수 교훈: content가 아니라 text).
"""

from __future__ import annotations

from typing import Any

try:
    from elasticsearch import Elasticsearch
except ImportError:  # pragma: no cover
    Elasticsearch = None  # type: ignore[assignment]

from .models import WikiCard

META_FIELDS = ["title^3", "title.std^3", "description^2", "tags^2", "type"]
_SOURCE_META = ["path", "title", "description", "tags", "type",
                "status", "updated"]


def _mapping(dims: int) -> dict[str, Any]:
    return {
        "settings": {
            "analysis": {
                "analyzer": {
                    "korean": {"type": "custom", "tokenizer": "nori_tokenizer"}
                }
            }
        },
        "mappings": {
            "properties": {
                "path": {"type": "keyword"},
                "title": {"type": "text", "analyzer": "korean",
                          "fields": {"std": {"type": "text"},
                                     "kw": {"type": "keyword"}}},
                "description": {"type": "text", "analyzer": "korean"},
                "tags": {"type": "keyword"},
                "type": {"type": "keyword"},
                "status": {"type": "keyword"},
                "updated": {"type": "date",
                            "format": "yyyy-MM-dd||strict_date_optional_time"},
                "text": {"type": "text", "analyzer": "korean"},
                "embedding": {"type": "dense_vector", "dims": dims,
                              "index": True, "similarity": "cosine"},
            }
        },
    }


class ESIndexer:
    def __init__(self, es_url: str, index: str, dims: int = 1024):
        if Elasticsearch is None:  # pragma: no cover
            raise ImportError(
                "elasticsearch 클라이언트가 필요합니다: pip install elasticsearch")
        self.es = Elasticsearch(es_url, request_timeout=30)
        self.index = index
        self.dims = dims

    # ---------- 인덱스 수명주기 ----------
    def ensure_index(self) -> None:
        if self.es.indices.exists(index=self.index):
            return
        body = _mapping(self.dims)
        try:
            self.es.indices.create(index=self.index, **body)
        except Exception:
            # nori 플러그인 미설치 환경 폴백
            body["settings"]["analysis"]["analyzer"]["korean"] = {
                "type": "standard"}
            self.es.indices.create(index=self.index, **body)

    def drop_index(self) -> None:
        if self.es.indices.exists(index=self.index):
            self.es.indices.delete(index=self.index)

    # ---------- 색인 ----------
    def index_card(self, card: WikiCard, rel_path: str,
                   embedding: list[float] | None = None) -> None:
        fm = card.frontmatter
        doc: dict[str, Any] = {
            "path": rel_path,
            "title": fm.title,
            "description": fm.description,
            "tags": fm.tags,
            "type": fm.type,
            "status": fm.status,
            "updated": fm.updated,
            "text": card.body,
        }
        if embedding is not None:
            doc["embedding"] = embedding
        self.es.index(index=self.index, id=rel_path, document=doc)

    def delete_card(self, rel_path: str) -> None:
        self.es.delete(index=self.index, id=rel_path, ignore=[404])

    def refresh(self) -> None:
        self.es.indices.refresh(index=self.index)

    # ---------- 1단계: 메타 검색 (본문 비노출) ----------
    def meta_search(self, query: str, k: int = 20,
                    active_only: bool = True) -> list[dict[str, Any]]:
        q: dict[str, Any] = {
            "bool": {
                "must": [{"multi_match": {"query": query,
                                          "fields": META_FIELDS,
                                          "type": "best_fields"}}]
            }
        }
        if active_only:
            q["bool"]["filter"] = [{"term": {"status": "active"}}]
        res = self.es.search(index=self.index, query=q, size=k,
                             source=_SOURCE_META)
        return _hits(res)

    # ---------- 2단계: 본문 하이브리드 (BM25 + kNN, 수동 RRF) ----------
    def body_bm25(self, query: str, k: int = 20) -> list[dict[str, Any]]:
        q = {"multi_match": {"query": query,
                             "fields": ["text", "title^2", "description"]}}
        res = self.es.search(index=self.index, query=q, size=k,
                             source=_SOURCE_META)
        return _hits(res)

    def knn(self, vector: list[float], k: int = 20) -> list[dict[str, Any]]:
        res = self.es.search(
            index=self.index,
            knn={"field": "embedding", "query_vector": vector,
                 "k": k, "num_candidates": max(50, k * 5)},
            size=k, source=_SOURCE_META)
        return _hits(res)

    def hybrid_search(self, query: str, vector: list[float] | None,
                      k: int = 10) -> list[dict[str, Any]]:
        bm = self.body_bm25(query, k=max(20, k * 3))
        if vector is None:
            return bm[:k]
        kn = self.knn(vector, k=max(20, k * 3))
        return rrf_fuse([bm, kn], k=k)

    # ---------- 유사 카드 (lint용) ----------
    def more_like(self, rel_path: str, text: str,
                  vector: list[float] | None, k: int = 5
                  ) -> list[dict[str, Any]]:
        if vector is not None:
            hits = self.knn(vector, k=k + 1)
        else:
            hits = self.body_bm25(text[:512], k=k + 1)
        return [h for h in hits if h.get("path") != rel_path][:k]


def rrf_fuse(result_lists: list[list[dict[str, Any]]], k: int = 10,
             c: int = 60) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion (하네스와 동일 방식의 수동 융합)."""
    scores: dict[str, float] = {}
    keep: dict[str, dict[str, Any]] = {}
    for results in result_lists:
        for rank, hit in enumerate(results):
            key = hit["path"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (c + rank + 1)
            keep.setdefault(key, hit)
    ordered = sorted(scores.items(), key=lambda t: -t[1])
    out = []
    for key, s in ordered[:k]:
        h = dict(keep[key])
        h["_rrf"] = round(s, 6)
        out.append(h)
    return out


def _hits(res: Any) -> list[dict[str, Any]]:
    out = []
    for h in res.get("hits", {}).get("hits", []):
        d = dict(h.get("_source", {}))
        d["_score"] = h.get("_score")
        out.append(d)
    return out

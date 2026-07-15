"""
gemma-rag-harness / retrieval/indexer.py

코퍼스를 받아 청킹 → 임베딩 → ES 색인하는 파이프라인.

  raw 문서들 ──chunk──▶ 청크들 ──embed(bge-m3)──▶ 벡터들 ──▶ ESStore.index

임베딩 모델: BAAI/bge-m3 (다국어 SOTA, 한/영 혼용 코퍼스에 적합, 1024차원)
질의 시에도 같은 모델·같은 normalize를 써야 kNN이 의미를 가진다.

의존: pip install sentence-transformers tiktoken
"""

from __future__ import annotations

import hashlib

import tiktoken
from sentence_transformers import SentenceTransformer

from retrieval.es_store import ESStore
from retrieval.vector_store import Doc

# 청킹 파라미터: 토큰 기준 (문자 기준보다 LLM 컨텍스트 예산과 일관됨)
CHUNK_TOKENS = 400
CHUNK_OVERLAP = 80

_ENC = tiktoken.get_encoding("cl100k_base")
_MODEL: SentenceTransformer | None = None


def _model() -> SentenceTransformer:
    """임베딩 모델 지연 로딩(임포트 시 GPU 점유 방지)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("BAAI/bge-m3")
    return _MODEL


def chunk(text: str, source: str) -> list[Doc]:
    """토큰 단위 슬라이딩 윈도우 청킹. id는 내용 해시로 안정적으로 생성."""
    tokens = _ENC.encode(text)
    out: list[Doc] = []
    step = CHUNK_TOKENS - CHUNK_OVERLAP
    for start in range(0, len(tokens), step):
        piece = tokens[start:start + CHUNK_TOKENS]
        if not piece:
            break
        chunk_text = _ENC.decode(piece)
        cid = hashlib.sha1(f"{source}:{start}:{chunk_text}".encode()).hexdigest()[:16]
        out.append(Doc(id=cid, text=chunk_text, score=0.0, source=source))
        if start + CHUNK_TOKENS >= len(tokens):
            break
    return out


def embed(texts: list[str]) -> list[list[float]]:
    """normalize_embeddings=True → cosine 유사도와 정합."""
    vecs = _model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]


def embed_query(query: str) -> list[float]:
    """질의 임베딩(색인과 동일 모델/정규화). graph.py의 retrieve가 호출."""
    return embed([query])[0]


def index_corpus(corpus: dict[str, str], store: ESStore | None = None) -> int:
    """
    corpus: {source_name: full_text} 매핑.
    반환: 색인된 청크 수.
    """
    store = store or ESStore()
    all_docs: list[Doc] = []
    for source, text in corpus.items():
        all_docs.extend(chunk(text, source))

    vectors = embed([d.text for d in all_docs])
    store.index(all_docs, vectors=vectors)
    return len(all_docs)


if __name__ == "__main__":
    # 최소 동작 확인용 샘플 코퍼스
    sample = {
        "es-hybrid.md": (
            "Elasticsearch 8.x는 retriever 문법으로 BM25와 kNN을 결합한다. "
            "rrf 리트리버는 두 결과의 순위를 RRF로 융합하며 점수 스케일 보정이 필요 없다. "
            "dense_vector 필드는 cosine 유사도와 kNN 검색을 지원한다."
        ),
    }
    n = index_corpus(sample)
    print(f"indexed {n} chunks")

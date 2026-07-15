"""
gemma-rag-harness / retrieval/vector_store.py

검색 백엔드 추상 인터페이스.
공고가 Elasticsearch와 Cosmos DB(vCore)를 둘 다 언급하므로,
검색 구현을 이 인터페이스 뒤에 숨겨 교체 가능하게 둔다.
하네스(graph.py)는 이 타입에만 의존하고 구체 구현(ES/Cosmos)을 모른다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Doc:
    """검색 결과 단위. id는 골든셋 평가(Recall@k, MRR)에서 정답 매칭에 쓴다."""
    id: str
    text: str
    score: float
    source: str = ""          # 원문 출처(파일명/URL 등), 인용 표기에 사용


class VectorStore(Protocol):
    """검색 백엔드가 구현해야 하는 계약."""

    def index(self, docs: list[Doc]) -> None:
        """문서를 색인한다. (임베딩은 indexer에서 채워 넘김)"""
        ...

    def hybrid_search(self, query: str, query_vector: list[float], k: int = 5) -> list[Doc]:
        """BM25(키워드) + dense(kNN) 결과를 RRF로 융합해 상위 k개를 반환."""
        ...

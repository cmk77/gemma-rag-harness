"""임베딩 모듈 (bge-m3, 하네스와 동일 모델).

sentence-transformers가 없거나 로드 실패 시 None을 반환해
검색이 BM25-only로 자동 강등되도록 설계 (집 노트북 경로 B 호환).
"""

from __future__ import annotations

from typing import Optional


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self._model = None
        self._failed = False

    def _load(self):
        if self._model is not None or self._failed:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        except Exception:
            self._failed = True

    @property
    def available(self) -> bool:
        self._load()
        return self._model is not None

    def encode(self, texts: list[str]) -> Optional[list[list[float]]]:
        self._load()
        if self._model is None:
            return None
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]

    def encode_one(self, text: str) -> Optional[list[float]]:
        out = self.encode([text])
        return out[0] if out else None

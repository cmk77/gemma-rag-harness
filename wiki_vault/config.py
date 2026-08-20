"""설정 모듈.

기존 gemma-rag-harness 컨벤션을 그대로 따른다:
  - LLM: OpenAI 호환 엔드포인트 (VLLM_URL / LLM_MODEL 두 변수만으로
    vLLM(워크스테이션) <-> Ollama(집 노트북, 경로 B) 전환)
  - ES:  ES_URL + 전용 인덱스(WIKI_ES_INDEX), 본문 필드명은 `text`
  - 임베딩: bge-m3 (1024차원)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str) -> str:
    v = os.environ.get(key, "").strip()
    return v if v else default


@dataclass
class WikiSettings:
    # ---- 볼트 경로 ----
    wiki_root: Path = field(
        default_factory=lambda: Path(_env("WIKI_VAULT_ROOT", "./vault_wiki")))
    # 모선(메인 볼트). 비우면 폴백 비활성.
    main_root: Path | None = field(
        default_factory=lambda: (
            Path(p) if (p := _env("MAIN_VAULT_ROOT", "")) else None))

    # ---- LLM (OpenAI 호환) ----
    llm_base_url: str = field(
        default_factory=lambda: _env("VLLM_URL", "http://localhost:8000/v1"))
    llm_model: str = field(
        default_factory=lambda: _env("LLM_MODEL", "gemma-4-E4B-it"))
    llm_timeout: float = field(
        default_factory=lambda: float(_env("LLM_TIMEOUT", "120")))

    # ---- Elasticsearch ----
    es_url: str = field(
        default_factory=lambda: _env("ES_URL", "http://localhost:9200"))
    es_index: str = field(
        default_factory=lambda: _env("WIKI_ES_INDEX", "gemma_rag_wiki_vault"))

    # ---- 임베딩 ----
    embed_model: str = field(
        default_factory=lambda: _env("EMBED_MODEL", "BAAI/bge-m3"))
    embed_dims: int = field(
        default_factory=lambda: int(_env("EMBED_DIMS", "1024")))
    use_embeddings: bool = field(
        default_factory=lambda: _env("WIKI_USE_EMBEDDINGS", "1") != "0")

    # ---- 검색/토큰 예산 (Progressive Disclosure) ----
    meta_k: int = field(default_factory=lambda: int(_env("WIKI_META_K", "20")))
    top_k: int = field(default_factory=lambda: int(_env("WIKI_TOP_K", "5")))
    # 하네스 ADR-0001 교훈 반영: 재검색 시 폭 확장
    wide_k: int = field(default_factory=lambda: int(_env("WIKI_WIDE_K", "10")))
    body_char_budget: int = field(
        default_factory=lambda: int(_env("WIKI_BODY_BUDGET", "4000")))
    max_retries: int = field(
        default_factory=lambda: int(_env("WIKI_MAX_RETRIES", "1")))

    def ensure_dirs(self) -> None:
        self.wiki_root.mkdir(parents=True, exist_ok=True)

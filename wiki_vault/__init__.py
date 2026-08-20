"""wiki_vault — LLM Wiki 지식 볼트 모듈 (gemma-rag-harness 애드온).

강연 'AI 에이전트를 위한 지식 베이스 구축(LLM Wiki)'의 핵심 개념을
기존 하네스 스택(vLLM OpenAI 호환, Elasticsearch nori/BM25+kNN/RRF,
bge-m3, LangGraph)에 맞춰 모듈화한 패키지.

개념 → 모듈 매핑
  듀얼 볼트(모선/위성)               -> vault.VaultManager
  YAML 메타 + Progressive Disclosure -> models.WikiCard / indexer.ESIndexer
  Ingest (inbox -> 카드)             -> ingest.IngestPipeline
  Query (BM25 -> hybrid -> 검증)     -> query.QueryEngine
  Lint  (구지식 최신화/감사)          -> lint.LintEngine
  corecontext / rules                -> cli onboard

무거운 의존성(elasticsearch, sentence-transformers)은 실제 사용 시점에만
로드되도록 지연 임포트한다.
"""

from typing import Any

from .config import WikiSettings
from .models import CardType, Frontmatter, WikiCard
from .vault import VaultManager

__all__ = [
    "WikiSettings", "CardType", "Frontmatter", "WikiCard", "VaultManager",
    "LLMClient", "ESIndexer", "IngestPipeline", "QueryEngine", "LintEngine",
]
__version__ = "0.1.0"

_LAZY = {
    "LLMClient": ("wiki_vault.llm", "LLMClient"),
    "ESIndexer": ("wiki_vault.indexer", "ESIndexer"),
    "IngestPipeline": ("wiki_vault.ingest", "IngestPipeline"),
    "QueryEngine": ("wiki_vault.query", "QueryEngine"),
    "LintEngine": ("wiki_vault.lint", "LintEngine"),
}


def __getattr__(name: str) -> Any:  # PEP 562 지연 임포트
    if name in _LAZY:
        import importlib
        mod, attr = _LAZY[name]
        return getattr(importlib.import_module(mod), attr)
    raise AttributeError(f"module 'wiki_vault' has no attribute {name!r}")

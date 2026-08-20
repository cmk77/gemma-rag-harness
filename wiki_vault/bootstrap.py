"""컴포넌트 조립 팩토리 — CLI, 하네스, 웹UI가 공용으로 쓰는 진입점."""

from __future__ import annotations

from dataclasses import dataclass

from .config import WikiSettings
from .embeddings import Embedder
from .ingest import IngestPipeline
from .lint import LintEngine
from .llm import LLMClient
from .query import QueryEngine
from .vault import VaultManager


@dataclass
class WikiApp:
    settings: WikiSettings
    vault: VaultManager
    llm: LLMClient
    embedder: Embedder | None
    indexer: "object | None"
    ingest: IngestPipeline
    query: QueryEngine
    lint: LintEngine


def build_app(settings: WikiSettings | None = None,
              with_es: bool = True) -> WikiApp:
    s = settings or WikiSettings()
    s.ensure_dirs()

    vault = VaultManager(s.wiki_root, s.main_root)
    vault.scaffold()
    llm = LLMClient(s.llm_base_url, s.llm_model, timeout=s.llm_timeout)
    embedder = Embedder(s.embed_model) if s.use_embeddings else None

    indexer = None
    if with_es:
        from .indexer import ESIndexer
        indexer = ESIndexer(s.es_url, s.es_index, dims=s.embed_dims)
        indexer.ensure_index()

    lint = LintEngine(vault, llm, indexer=indexer, embedder=embedder)
    ingest = IngestPipeline(vault, llm, indexer=indexer,
                            embedder=embedder, lint=lint)
    query = QueryEngine(s, vault, indexer, llm, embedder=embedder)
    return WikiApp(settings=s, vault=vault, llm=llm, embedder=embedder,
                   indexer=indexer, ingest=ingest, query=query, lint=lint)


def build_default_engine() -> QueryEngine:
    """하네스 ROUTER 통합용 간편 팩토리."""
    return build_app().query


def reindex_all(app: WikiApp) -> int:
    """볼트 전수 재색인 (마이그레이션/복구용)."""
    if app.indexer is None:
        raise RuntimeError("ES 비활성 상태에서는 재색인 불가")
    n = 0
    for card in app.vault.iter_cards():
        vec = None
        if app.embedder is not None and app.embedder.available:
            vec = app.embedder.encode_one(
                f"{card.frontmatter.title}\n{card.body[:3000]}")
        app.indexer.index_card(card, app.vault.rel(card.path), vec)
        n += 1
    app.indexer.refresh()
    return n

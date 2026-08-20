"""Ingest — 로우 데이터를 위키 카드로 '소화·흡수'하는 파이프라인.

강연 포인트 반영:
  - 인박스(0_inbox) 버퍼에 모아두었다가 일괄 인제스트
  - AI가 성격이 비슷한 캡처본을 스스로 묶어 그룹 인제스트 제안
  - 사용자가 캡처 시 적어둔 #목적 해시태그를 카드 태그로 승계
    (휴먼 바이어스/취향 제공 — 가장 중요한 사용자 주도 행위)
  - 논문은 전문 인제스트가 토큰 낭비 -> 초록 모드(abstract_only) 지원
  - 카드 메타에 작성 모델(model:)을 기록해 모델별 품질 추적
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .llm import LLMClient
from .models import CardType, Frontmatter, WikiCard
from .vault import VaultManager

_HASHTAG_RE = re.compile(r"#([\w가-힣/·-]+)")

_GROUP_SYS = """너는 지식 관리 사서다. 인박스에 캡처된 문서 목록을 보고,
주제가 밀접한 것끼리 그룹으로 묶어라. 혼자인 문서는 단독 그룹.
출력 스키마: {"groups": [{"topic": str, "reason": str, "item_ids": [int]}]}"""

_INGEST_SYS = """너는 개인 지식 볼트의 위키 카드 작성자다.
주어진 로우 데이터를 소화해 아래 규칙으로 위키 카드(들)를 만든다.

규칙:
1. 카드 종류는 concept(개념) 또는 entity(제품/조직/인물/도구).
2. title과 description은 에이전트 파싱 효율을 위해 영어로 쓴다.
   description은 카드를 열지 않고도 내용을 판단할 수 있는 1~2문장.
3. body는 한국어로, 사람이 읽기 좋은 짧은 섹션 구조(## 개요, ## 핵심 사실,
   ## 맥락/코멘터리)로 쓴다. 원문 통짜 복사 금지 — 반드시 소화된 요약.
4. 사용자 목적 태그(purpose_tags)가 있으면 tags에 그대로 포함한다.
5. 관련될 법한 다른 카드 제목이 있으면 links에 넣는다(없으면 빈 배열).
6. abstract_only=true면 초록/핵심 주장 수준까지만 요약하고,
   본문에 '원문 링크 참조'를 명시한다.

출력 스키마(JSON):
{"cards": [{"type": "concept|entity", "title": str, "description": str,
            "tags": [str], "links": [str], "body": str}]}"""


@dataclass
class InboxItem:
    id: int
    path: Path
    title: str
    purpose_tags: list[str]
    preview: str


@dataclass
class IngestResult:
    cards: list[WikiCard] = field(default_factory=list)
    archived_sources: list[Path] = field(default_factory=list)


class IngestPipeline:
    def __init__(self, vault: VaultManager, llm: LLMClient,
                 indexer=None, embedder=None, lint=None):
        self.vault = vault
        self.llm = llm
        self.indexer = indexer
        self.embedder = embedder
        self.lint = lint  # LintEngine (선택) — 인제스트 후 구지식 최신화 훅

    # ---------- 1) 인박스 스캔/그룹핑 ----------
    def scan_inbox(self) -> list[InboxItem]:
        items = []
        for i, p in enumerate(self.vault.inbox_items()):
            text = p.read_text(encoding="utf-8", errors="ignore")
            tags = sorted(set(_HASHTAG_RE.findall(text[:2000])))
            items.append(InboxItem(
                id=i, path=p, title=p.stem, purpose_tags=tags,
                preview=_strip(text)[:400]))
        return items

    def group_items(self, items: list[InboxItem]) -> list[dict]:
        """/inbox 스킬: 성격이 비슷한 캡처본을 AI가 묶어 제안."""
        if not items:
            return []
        if len(items) == 1:
            return [{"topic": items[0].title, "reason": "single item",
                     "item_ids": [0]}]
        listing = "\n".join(
            f"[{it.id}] {it.title} | tags={it.purpose_tags} | {it.preview}"
            for it in items)
        try:
            out = self.llm.chat_json(_GROUP_SYS, listing)
            groups = out.get("groups", []) if isinstance(out, dict) else []
        except Exception:
            groups = []
        if not groups:  # LLM 실패 시 각자 단독 그룹
            groups = [{"topic": it.title, "reason": "fallback",
                       "item_ids": [it.id]} for it in items]
        return groups

    # ---------- 2) 그룹 인제스트 (지식 병합) ----------
    def ingest_group(self, items: list[InboxItem],
                     abstract_only: bool = False,
                     extra_context: str = "") -> IngestResult:
        payload = {
            "abstract_only": abstract_only,
            "user_context": self.vault.system_context()[:1500],
            "commentary": extra_context,  # 사용자 코멘터리(중요)
            "items": [{
                "title": it.title,
                "purpose_tags": it.purpose_tags,
                "source_path": str(it.path.name),
                "text": _read_for_ingest(it.path, abstract_only),
            } for it in items],
        }
        out = self.llm.chat_json(
            _INGEST_SYS, json.dumps(payload, ensure_ascii=False))
        raw_cards = out.get("cards", []) if isinstance(out, dict) else []

        result = IngestResult()
        source_names = ", ".join(it.path.name for it in items)
        for rc in raw_cards:
            fm = Frontmatter(
                title=str(rc.get("title", "untitled")),
                description=str(rc.get("description", "")),
                type=(rc.get("type") if rc.get("type")
                      in (CardType.CONCEPT.value, CardType.ENTITY.value)
                      else CardType.CONCEPT.value),
                tags=list(dict.fromkeys(
                    [*(rc.get("tags") or []),
                     *(t for it in items for t in it.purpose_tags)])),
                links=list(rc.get("links") or []),
                model=self.llm.model,           # 작성 모델 출처
                source=source_names,            # 로우 소스 추적
            )
            card = WikiCard(frontmatter=fm, body=str(rc.get("body", "")))
            path = self.vault.save_card(card)
            self._index(card)
            result.cards.append(card)
            _ = path

        # 로우 소스는 인박스에서 90_raw로 이동 (인제스트 후 정리)
        for it in items:
            result.archived_sources.append(
                self.vault.archive_inbox_item(it.path))

        # Lint 훅: 새 지식 유입 -> 관련 구지식 최신화
        if self.lint is not None and result.cards:
            try:
                self.lint.on_new_cards(result.cards)
            except Exception:
                pass
        return result

    def ingest_all(self, abstract_only: bool = False) -> list[IngestResult]:
        items = self.scan_inbox()
        by_id = {it.id: it for it in items}
        results = []
        for g in self.group_items(items):
            members = [by_id[i] for i in g.get("item_ids", []) if i in by_id]
            if members:
                results.append(self.ingest_group(
                    members, abstract_only=abstract_only))
        return results

    # ---------- 내부 ----------
    def _index(self, card: WikiCard) -> None:
        if self.indexer is None or card.path is None:
            return
        vec = None
        if self.embedder is not None and self.embedder.available:
            vec = self.embedder.encode_one(
                f"{card.frontmatter.title}\n{card.frontmatter.description}\n"
                f"{card.body[:3000]}")
        self.indexer.index_card(card, self.vault.rel(card.path), vec)


def _read_for_ingest(path: Path, abstract_only: bool,
                     full_budget: int = 12000,
                     abstract_budget: int = 3000) -> str:
    text = _strip(path.read_text(encoding="utf-8", errors="ignore"))
    return text[:abstract_budget if abstract_only else full_budget]


def _strip(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)      # 단순 HTML 제거
    return re.sub(r"[ \t]+", " ", text).strip()

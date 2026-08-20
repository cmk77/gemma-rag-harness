"""Lint — 새 지식이 들어올 때 구지식을 최신화하는 유지보수 엔진.

강연의 Audit / Verify / Lint 3종 스킬 대응:
  audit()        : 볼트 전수 감사 — 깨진 링크, 프론트매터 누락, 스테일 카드
  verify(card)   : 개별 카드 사실성 재점검 (LLM)
  on_new_cards() : 인제스트 훅 — 관련 카드 탐색(links + kNN) 후
                   outdated 판정/패치/교차링크
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field

from .llm import LLMClient
from .models import WikiCard
from .vault import VaultManager

_COMPARE_SYS = """새 카드와 기존 카드를 비교하라. 기존 카드 관점에서 판단:
- "none"     : 영향 없음
- "link"     : 서로 관련 — 교차 링크만 추가
- "update"   : 기존 카드 일부가 구식 — patch에 갱신 요지(한국어 2~4줄)
- "outdated" : 기존 카드 전체가 새 카드로 대체됨
출력: {"action": "none|link|update|outdated", "patch": str, "reason": str}"""

_VERIFY_CARD_SYS = """카드 본문을 검토해 내부 모순, 근거 없는 단정,
날짜/버전 표기 누락을 지적하라.
출력: {"ok": bool, "issues": [str]}"""


@dataclass
class LintReport:
    updated: list[str] = field(default_factory=list)
    outdated: list[str] = field(default_factory=list)
    linked: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, indent=2)


class LintEngine:
    def __init__(self, vault: VaultManager, llm: LLMClient,
                 indexer=None, embedder=None, stale_days: int = 180):
        self.vault = vault
        self.llm = llm
        self.indexer = indexer
        self.embedder = embedder
        self.stale_days = stale_days

    # ================= 인제스트 훅 =================
    def on_new_cards(self, new_cards: list[WikiCard]) -> LintReport:
        report = LintReport()
        for new in new_cards:
            for old in self._related(new):
                self._reconcile(new, old, report)
        if self.indexer is not None:
            try:
                self.indexer.refresh()
            except Exception:
                pass
        return report

    def _related(self, card: WikiCard, k: int = 5) -> list[WikiCard]:
        """관련 카드 = 명시 링크(제목·경로) + 의미 유사(kNN/BM25) 상위 k."""
        found: dict[str, WikiCard] = {}
        # 1) 프론트매터 links + [[위키링크]] + 변환된 md 링크
        refs = (set(card.frontmatter.links) | set(card.wikilinks())
                | set(card.mdlinks()))
        for ref in refs:
            hit = self._resolve(ref)
            if hit is not None:
                found[self.vault.rel(hit.path)] = hit
        # 2) 색인 유사도
        if self.indexer is not None and card.path is not None:
            vec = None
            if self.embedder is not None and self.embedder.available:
                vec = self.embedder.encode_one(card.body[:2000])
            for h in self.indexer.more_like(
                    self.vault.rel(card.path), card.body, vec, k=k):
                if h["path"] not in found:
                    try:
                        found[h["path"]] = self.vault.load_card(h["path"])
                    except Exception:
                        pass
        # 자기 자신 제외
        self_rel = self.vault.rel(card.path) if card.path else None
        return [c for rel, c in found.items() if rel != self_rel]

    def _resolve(self, ref: str) -> WikiCard | None:
        """ref가 상대경로(.md)면 직접, 아니면 제목으로 위키 볼트 내 탐색."""
        if ref.endswith(".md"):
            p = self.vault.wiki_root / ref
            if p.exists():
                try:
                    return self.vault.load_card(p)
                except Exception:
                    return None
            return None
        hit = self.vault._find_note(ref)  # noqa: SLF001 (내부 재사용)
        if hit and self.vault.wiki_root in hit.parents:
            try:
                return self.vault.load_card(hit)
            except Exception:
                return None
        return None

    def _reconcile(self, new: WikiCard, old: WikiCard,
                   report: LintReport) -> None:
        try:
            out = self.llm.chat_json(_COMPARE_SYS, json.dumps({
                "new": {"title": new.frontmatter.title,
                        "description": new.frontmatter.description,
                        "body": new.body[:3000]},
                "old": {"title": old.frontmatter.title,
                        "description": old.frontmatter.description,
                        "updated": old.frontmatter.updated,
                        "body": old.body[:3000]},
            }, ensure_ascii=False))
        except Exception:
            return
        action = out.get("action", "none")
        rel_old = self.vault.rel(old.path) if old.path else old.frontmatter.title
        rel_new = self.vault.rel(new.path) if new.path else new.frontmatter.title
        today = _dt.date.today().isoformat()

        if action == "link":
            self._cross_link(new, old)
            report.linked.append(f"{rel_new} <-> {rel_old}")
        elif action == "update":
            old.body += (f"\n\n## Lint 갱신 ({today})\n"
                         f"{out.get('patch', '').strip()}\n"
                         f"(근거: [{new.frontmatter.title}]({rel_new}))")
            old.frontmatter.updated = today
            self._save_and_reindex(old)
            self._cross_link(new, old)
            report.updated.append(rel_old)
        elif action == "outdated":
            old.frontmatter.status = "outdated"
            old.frontmatter.updated = today
            old.body = (f"> [!warning] {today} 이후 구식 — "
                        f"[{new.frontmatter.title}]({rel_new}) 참고\n\n"
                        + old.body)
            self._save_and_reindex(old)
            report.outdated.append(rel_old)

    def _cross_link(self, a: WikiCard, b: WikiCard) -> None:
        for src, dst in ((a, b), (b, a)):
            rel = self.vault.rel(dst.path) if dst.path else ""
            if rel and rel not in src.frontmatter.links:
                src.frontmatter.links.append(rel)
                self._save_and_reindex(src)

    def _save_and_reindex(self, card: WikiCard) -> None:
        self.vault.save_card(card)
        if self.indexer is not None and card.path is not None:
            vec = None
            if self.embedder is not None and self.embedder.available:
                vec = self.embedder.encode_one(card.body[:3000])
            self.indexer.index_card(card, self.vault.rel(card.path), vec)

    # ================= 감사(audit) =================
    def audit(self) -> LintReport:
        report = LintReport()
        today = _dt.date.today()
        for card in self.vault.iter_cards():
            rel = self.vault.rel(card.path) if card.path else "?"
            fm = card.frontmatter
            if not fm.title or not fm.description:
                report.issues.append(f"{rel}: 프론트매터 title/description 누락")
            for name in card.wikilinks():
                if self.vault._find_note(name) is None:  # noqa: SLF001
                    report.issues.append(f"{rel}: 깨진 링크 [[{name}]]")
            try:
                upd = _dt.date.fromisoformat(str(fm.updated)[:10])
                if (fm.status == "active"
                        and (today - upd).days > self.stale_days):
                    report.issues.append(
                        f"{rel}: {self.stale_days}일 이상 미갱신(stale)")
            except ValueError:
                report.issues.append(f"{rel}: updated 날짜 형식 오류")
        return report

    # ================= 개별 검증(verify) =================
    def verify(self, rel_path: str) -> dict:
        card = self.vault.load_card(rel_path)
        return self.llm.chat_json(
            _VERIFY_CARD_SYS,
            f"title: {card.frontmatter.title}\n\n{card.body[:6000]}")

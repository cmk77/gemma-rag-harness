"""듀얼 볼트 관리자 (모선/위성 분리 철학).

  - 위키 볼트(위성): AI가 캡처/요약한 지식. 이 모듈이 쓰기 권한을 가짐.
  - 메인 볼트(모선): 사용자가 직접 체화해 쓴 지식. 이 모듈은 읽기 전용
    폴백 스캔만 수행한다 (정보 오염 방지 — 강연자의 볼트 분리 철학).

마크다운 파일이 소스오브트루스이고 Elasticsearch는 색인 사본이다.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Iterator

from .models import CardType, WikiCard

# 폴더 레이아웃 (지식 발전 상태 기준)
FOLDER_BY_TYPE = {
    CardType.CONCEPT.value: "10_concepts",
    CardType.ENTITY.value: "20_entities",
    CardType.NOTE.value: "30_notes",
    CardType.RAW.value: "90_raw",
}
INBOX_DIR = "0_inbox"          # 인제스트 전 버퍼 (강연의 0.2 인박스)
SYSTEM_DIR = "_system"         # corecontext.md / rules/
CORECONTEXT = "corecontext.md"
RULES_DIR = "rules"

_INGESTIBLE = {".md", ".txt", ".html"}


class VaultManager:
    def __init__(self, wiki_root: Path, main_root: Path | None = None):
        self.wiki_root = Path(wiki_root)
        self.main_root = Path(main_root) if main_root else None

    # ---------- 초기화 ----------
    def scaffold(self) -> None:
        for d in [INBOX_DIR, SYSTEM_DIR, f"{SYSTEM_DIR}/{RULES_DIR}",
                  *FOLDER_BY_TYPE.values()]:
            (self.wiki_root / d).mkdir(parents=True, exist_ok=True)

    # ---------- 카드 IO ----------
    def card_path(self, card: WikiCard) -> Path:
        folder = FOLDER_BY_TYPE.get(card.frontmatter.type, "30_notes")
        name = _slug(card.frontmatter.title) or "untitled"
        return self.wiki_root / folder / f"{name}.md"

    def save_card(self, card: WikiCard) -> Path:
        path = card.path if card.path else self.card_path(card)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 메인 볼트 연동 시 위키링크 -> 마크다운 링크 변환 (온보딩 기능)
        card.body = self.convert_wikilinks(card.body)
        path.write_text(card.to_markdown(), encoding="utf-8")
        card.path = path
        return path

    def load_card(self, path: Path | str) -> WikiCard:
        p = Path(path)
        if not p.is_absolute():
            p = self.wiki_root / p
        return WikiCard.from_markdown(p.read_text(encoding="utf-8"), path=p)

    def iter_cards(self) -> Iterator[WikiCard]:
        for folder in FOLDER_BY_TYPE.values():
            root = self.wiki_root / folder
            if not root.exists():
                continue
            for p in sorted(root.rglob("*.md")):
                try:
                    yield self.load_card(p)
                except Exception:
                    continue

    def rel(self, path: Path) -> str:
        try:
            return str(Path(path).relative_to(self.wiki_root))
        except ValueError:
            return str(path)

    # ---------- 인박스 ----------
    @property
    def inbox(self) -> Path:
        return self.wiki_root / INBOX_DIR

    def inbox_items(self) -> list[Path]:
        if not self.inbox.exists():
            return []
        return sorted(p for p in self.inbox.iterdir()
                      if p.is_file() and p.suffix.lower() in _INGESTIBLE)

    def archive_inbox_item(self, path: Path) -> Path:
        """인제스트 완료된 로우 소스는 90_raw로 이동 (원본 링크 보존)."""
        dest = self.wiki_root / FOLDER_BY_TYPE[CardType.RAW.value] / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dest))
        return dest

    # ---------- 시스템 파일 (corecontext / rules) ----------
    def system_context(self) -> str:
        """세션 시작 시 LLM 시스템 프롬프트에 주입할 컨텍스트.

        강연의 .claude rules 개념: rules가 가장 먼저, 강제적으로 읽힌다.
        """
        parts: list[str] = []
        rules_dir = self.wiki_root / SYSTEM_DIR / RULES_DIR
        if rules_dir.exists():
            for p in sorted(rules_dir.glob("*.md")):
                parts.append(f"<rule name='{p.stem}'>\n"
                             f"{p.read_text(encoding='utf-8').strip()}\n</rule>")
        core = self.wiki_root / SYSTEM_DIR / CORECONTEXT
        if core.exists():
            parts.append(f"<corecontext>\n"
                         f"{core.read_text(encoding='utf-8').strip()}\n"
                         f"</corecontext>")
        return "\n\n".join(parts)

    def write_system_files(self, corecontext: str, rules: str) -> None:
        sysdir = self.wiki_root / SYSTEM_DIR
        (sysdir / RULES_DIR).mkdir(parents=True, exist_ok=True)
        (sysdir / CORECONTEXT).write_text(corecontext, encoding="utf-8")
        (sysdir / RULES_DIR / "00_wiki_rules.md").write_text(
            rules, encoding="utf-8")

    # ---------- 링크 변환 ----------
    def convert_wikilinks(self, body: str) -> str:
        """[[이름]] -> 마크다운 링크. 대상 우선순위: 위키 볼트 > 메인 볼트.

        메인 볼트 파일을 가리키면 절대경로 링크로 남겨 외부 볼트와 연결.
        대상이 없으면 원문 유지(추후 lint audit에서 보고).
        """
        def _sub(m: re.Match) -> str:
            name = m.group(1).strip()
            hit = self._find_note(name)
            if hit is None:
                return m.group(0)
            if self.wiki_root in hit.parents or hit == self.wiki_root:
                return f"[{name}]({self.rel(hit)})"
            return f"[{name}]({hit})"

        return re.sub(r"\[\[([^\]|#]+)(?:\|[^\]]*)?\]\]", _sub, body)

    def _find_note(self, name: str) -> Path | None:
        target = f"{_slug(name)}.md"
        raw = f"{name}.md"
        for root in filter(None, [self.wiki_root, self.main_root]):
            for cand in (raw, target):
                hits = list(Path(root).rglob(cand))
                if hits:
                    return hits[0]
        return None

    # ---------- 모선 폴백 스캔 (읽기 전용) ----------
    def scan_main_vault(self, keywords: list[str], limit: int = 3,
                        snippet_chars: int = 1200) -> list[tuple[Path, str]]:
        """위키 볼트 지식이 부족할 때 메인 볼트를 키워드로 훑는다."""
        if not self.main_root or not Path(self.main_root).exists():
            return []
        kws = [k.lower() for k in keywords if len(k) >= 2]
        scored: list[tuple[int, Path, str]] = []
        for p in Path(self.main_root).rglob("*.md"):
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            low = text.lower()
            score = sum(low.count(k) for k in kws)
            if score > 0:
                scored.append((score, p, text[:snippet_chars]))
        scored.sort(key=lambda t: -t[0])
        return [(p, s) for _, p, s in scored[:limit]]


def _slug(title: str) -> str:
    s = re.sub(r"[^\w\s가-힣-]", "", title).strip()
    return re.sub(r"\s+", "-", s)[:80]

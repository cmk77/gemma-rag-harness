"""위키 카드 데이터 모델.

강연 핵심: AI가 본문 전체를 열기 전에 YAML 메타데이터(제목/설명/태그)만
먼저 읽는 Progressive Disclosure. 따라서 카드는 항상
[프론트매터 블록] + [본문] 구조를 강제하고, 제목/설명은 에이전트
파싱 효율을 위해 가급적 영어로 작성한다(ingest 프롬프트에서 강제).
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise ImportError("pyyaml이 필요합니다: pip install pyyaml") from e

_FM_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.S)


class CardType(str, Enum):
    CONCEPT = "concept"   # 개념 카드
    ENTITY = "entity"     # 엔티티 카드 (인물/제품/조직 등)
    RAW = "raw"           # 인제스트 전 로우 머터리얼
    NOTE = "note"         # 기타 노트 (쿼리 산출물 등)


def _today() -> str:
    return _dt.date.today().isoformat()


@dataclass
class Frontmatter:
    """카드 메타데이터. 이 블록만으로 '열어볼 가치'를 판단할 수 있어야 한다."""

    title: str = ""
    description: str = ""            # 1~2문장 영어 요약 권장
    type: str = CardType.NOTE.value
    tags: list[str] = field(default_factory=list)   # '#purpose/...' 목적 태그 포함
    status: str = "active"           # active | outdated | superseded
    created: str = field(default_factory=_today)
    updated: str = field(default_factory=_today)
    model: str = ""                  # 작성 모델 출처 (예: gemma-4-E4B-it)
    source: str = ""                 # 원문 URL/경로
    links: list[str] = field(default_factory=list)  # 관련 카드 경로
    vault: str = "wiki"              # wiki(위성) | main(모선)
    extra: dict[str, Any] = field(default_factory=dict)  # 사용자 커스텀 속성

    _KNOWN = ("title", "description", "type", "tags", "status", "created",
              "updated", "model", "source", "links", "vault")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {k: getattr(self, k) for k in self._KNOWN}
        d = {k: v for k, v in d.items() if v not in ("", [], None)}
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Frontmatter":
        fm = cls()
        extra: dict[str, Any] = {}
        for k, v in (d or {}).items():
            if k in cls._KNOWN:
                setattr(fm, k, v if v is not None else getattr(fm, k))
            else:
                extra[k] = v
        if isinstance(fm.tags, str):
            fm.tags = [t.strip() for t in fm.tags.split(",") if t.strip()]
        if isinstance(fm.links, str):
            fm.links = [fm.links]
        fm.extra = extra
        return fm


@dataclass
class WikiCard:
    frontmatter: Frontmatter
    body: str = ""
    path: Path | None = None  # 볼트 루트 기준 상대경로 or 절대경로

    # ---------- 직렬화 ----------
    def to_markdown(self) -> str:
        fm_yaml = yaml.safe_dump(
            self.frontmatter.to_dict(), allow_unicode=True, sort_keys=False,
            default_flow_style=False).strip()
        return f"---\n{fm_yaml}\n---\n\n{self.body.strip()}\n"

    @classmethod
    def from_markdown(cls, text: str, path: Path | None = None) -> "WikiCard":
        m = _FM_RE.match(text)
        if m:
            try:
                data = yaml.safe_load(m.group(1)) or {}
            except yaml.YAMLError:
                data = {}
            body = text[m.end():]
        else:
            data, body = {}, text
        fm = Frontmatter.from_dict(data if isinstance(data, dict) else {})
        if not fm.title and path is not None:
            fm.title = path.stem
        return cls(frontmatter=fm, body=body.strip(), path=path)

    # ---------- Progressive Disclosure 뷰 ----------
    def meta_line(self) -> str:
        """1단계 노출용 한 줄 요약 (본문 미포함)."""
        fm = self.frontmatter
        tags = " ".join(fm.tags[:6])
        return (f"[{fm.type}/{fm.status}] {fm.title} — {fm.description} "
                f"{('(' + tags + ')') if tags else ''}").strip()

    def clipped_body(self, budget: int) -> str:
        """2단계 노출: 토큰 예산 내 본문."""
        b = self.body
        return b if len(b) <= budget else b[:budget] + "\n...[truncated]"

    def wikilinks(self) -> list[str]:
        """본문 내 옵시디언 [[링크]] 목록 (미변환 잔존분)."""
        return re.findall(r"\[\[([^\]|#]+)", self.body)

    def mdlinks(self) -> list[str]:
        """본문 내 마크다운 링크 대상(.md) — 위키링크 변환 결과 포함."""
        return re.findall(r"\]\(([^)\s]+\.md)\)", self.body)

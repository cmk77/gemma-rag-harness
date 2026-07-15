"""
wiki/schema.py — 위키형 코퍼스 파이프라인의 공통 스키마.

대규모(수천~수백만 문서) 전제의 핵심 설계:
  - Fact = 파이프라인의 원자 단위. (엔티티, 섹션, 사실문장, 출처, 문서해시)
    문서가 아니라 '사실' 단위로 관리해야 증분 갱신·중복 제거·②단계(트리플)
    승격이 전부 가능해진다. Fact는 사실상 (주어, 술어≈섹션, 목적어) 준-트리플.
  - doc_hash = 내용 해시. 재실행 시 내용이 같으면 스킵(멱등),
    바뀌면 재추출하고 옛 해시의 사실은 빌드 단계에서 자동 무효화.
  - SECTIONS = 고정 섹션 체계. "취득 인증·수상"(certification)과
    "기능·지원 방식"(authentication 등 제품 기능)을 **다른 섹션으로 강제 분리**
    → '인증' 중의성이 검색 파라미터가 아니라 데이터 구조 차원에서 풀린다.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass

# 엔티티 위키 페이지의 고정 섹션 체계.
# 새 도메인에 쓰려면 이 목록과 prompts/wiki_extract.md 의 분류 규칙만 바꾼다.
SECTIONS: tuple[str, ...] = (
    "개요",            # 무엇인지 한 줄 정의, 소속·목적
    "취득 인증·수상",   # GS인증, 표창, 어워드 등 '받은' 자격 (certification)
    "기능·지원 방식",   # 제공 기능, 지원 프로토콜/포맷, 인증 '방식'(OAuth2 등)
    "도입·실적",        # 도입 기관 수, 처리 건수, 레퍼런스
    "연혁",            # 시점이 있는 사건 (출시, 이전, 취득 시점)
    "관계",            # 다른 엔티티와의 관계 (개발사, 상위 제품군, 구성요소)
    "기타",            # 위에 안 들어가는 사실
)

_SECTION_SET = set(SECTIONS)


@dataclass(frozen=True)
class Fact:
    """추출된 사실 1건. JSONL 한 줄로 저장된다."""
    entity_raw: str      # 문서에 등장한 표기 그대로 (정규화 전)
    section: str         # SECTIONS 중 하나
    fact: str            # 짧은 사실 문장 (한 문장, 자기완결)
    source: str          # 출처 문서 파일명
    doc_hash: str        # 출처 문서의 내용 해시 (증분 무효화 키)

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "Fact":
        return Fact(
            entity_raw=d["entity_raw"], section=d.get("section", "기타"),
            fact=d["fact"], source=d.get("source", "?"), doc_hash=d.get("doc_hash", ""),
        )


def valid_section(name: str) -> str:
    """LLM이 낸 섹션명을 검증. 목록 밖이면 '기타'로 강등(스키마 오염 방지)."""
    name = (name or "").strip()
    return name if name in _SECTION_SET else "기타"


def doc_hash(text: str) -> str:
    """문서 내용 해시 — 증분 처리의 멱등성 키."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def normalize_name(name: str) -> str:
    """
    엔티티 표기 정규화 키. 'GATEWAY-A' / 'gateway-a' / 'GATEWAY-A.' 이
    같은 키로 접히도록: 유니코드 NFKC → 대문자 → 영숫자/한글만 남김.
    (수백만 규모에선 이 규칙 기반 1차 통합 뒤, 임베딩 군집으로 2차 통합 —
     wiki/resolve.py 참고)
    """
    s = unicodedata.normalize("NFKC", name).upper()
    return re.sub(r"[^0-9A-Z가-힣]", "", s)


def fact_key(entity_canon: str, section: str, fact: str) -> str:
    """사실 중복 제거 키 — 공백·구두점 차이는 같은 사실로 접는다."""
    body = re.sub(r"[\s\W]+", "", unicodedata.normalize("NFKC", fact)).upper()
    return hashlib.sha1(f"{entity_canon}|{section}|{body}".encode()).hexdigest()[:16]


def slugify(entity_canon: str) -> str:
    """엔티티명 → 위키 페이지 파일명. 한글 유지, 공백은 하이픈."""
    s = re.sub(r"[^\w가-힣\- ]", "", entity_canon).strip()
    return re.sub(r"\s+", "-", s) or "unknown"

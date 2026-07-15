"""
wiki/resolve.py — 엔티티 정규화(canonicalization).

수백만 문서에서 가장 어려운 문제가 이것이다: 같은 대상이
"GATEWAY-A" / "GATEWAY-A" / "gateway-a" / "PRODUCT-A APIM" 으로 흩어지면
위키 페이지도 흩어지고 ②단계 그래프의 노드도 쪼개진다.

3단 전략 (앞 단계가 싸고 뒤로 갈수록 비싸다):
  1) 규칙 정규화  : schema.normalize_name — 대소문자·공백·기호·전각 접기 (무료)
  2) 별칭 사전    : wiki_aliases.json — 사람이/운영이 확정한 매핑 (무료, 우선순위 최상)
  3) 임베딩 군집  : 남은 표기들을 bge-m3로 임베딩해 코사인 유사도 그리디 군집
                    (--embed-resolve 옵션). 수백만 규모에선 이 단계를
                    ANN(FAISS/ES kNN) + LLM 판별로 승격 — 인터페이스 동일.

canonical 표기 선정: 같은 군집에서 **가장 자주 등장한 원표기**를 대표로 쓴다
(빈도는 대규모에서 가장 견고한 신호).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from wiki.schema import normalize_name

ALIASES_PATH = Path("wiki_aliases.json")   # {"정규화키 또는 원표기": "canonical 표기"}
EMBED_SIM_THRESHOLD = 0.90


def load_aliases() -> dict[str, str]:
    if ALIASES_PATH.exists():
        raw = json.loads(ALIASES_PATH.read_text(encoding="utf-8"))
        # 키도 정규화해서 조회 일관성 확보
        return {normalize_name(k): v for k, v in raw.items()}
    return {}


def build_canonical_map(entity_counts: Counter, use_embedding: bool = False,
                        ) -> dict[str, str]:
    """
    {원표기: canonical 표기} 매핑 생성.
    entity_counts: Counter(원표기 → 등장 횟수)
    """
    aliases = load_aliases()

    # 1+2단계: 정규화 키로 그룹핑, 별칭 사전이 있으면 그 표기를 canonical로
    groups: dict[str, Counter] = {}
    for raw, cnt in entity_counts.items():
        groups.setdefault(normalize_name(raw), Counter())[raw] += cnt

    canon_of_key: dict[str, str] = {}
    for key, variants in groups.items():
        canon_of_key[key] = aliases.get(key) or variants.most_common(1)[0][0]

    # 3단계(선택): 정규화로도 안 접힌 유사 표기를 임베딩으로 군집
    if use_embedding and len(canon_of_key) > 1:
        canon_of_key = _merge_by_embedding(canon_of_key, groups)

    return {raw: canon_of_key[normalize_name(raw)] for raw in entity_counts}


def _merge_by_embedding(canon_of_key: dict[str, str],
                        groups: dict[str, Counter]) -> dict[str, str]:
    """빈도 내림차순 그리디 군집: 상위 표기가 흡수 주체가 된다."""
    from retrieval.indexer import embed  # bge-m3 재사용 (지연 임포트)

    keys = sorted(canon_of_key, key=lambda k: -sum(groups[k].values()))
    names = [canon_of_key[k] for k in keys]
    vecs = embed(names)

    def cos(a, b):
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb + 1e-9)

    absorbed: dict[str, str] = {}
    for i, ki in enumerate(keys):
        if ki in absorbed:
            continue
        for j in range(i + 1, len(keys)):
            kj = keys[j]
            if kj in absorbed:
                continue
            if cos(vecs[i], vecs[j]) >= EMBED_SIM_THRESHOLD:
                absorbed[kj] = ki                      # 저빈도가 고빈도에 흡수
                print(f"  [resolve/embed] '{canon_of_key[kj]}' → '{canon_of_key[ki]}'")
    return {k: canon_of_key[absorbed.get(k, k)] for k in canon_of_key}

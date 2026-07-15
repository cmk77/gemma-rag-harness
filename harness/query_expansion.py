"""
gemma-rag-harness / harness/query_expansion.py

재검색 루프에서 "검색 입력을 바꾸기" 위한 쿼리 확장 전략들.
모두 Gemma 4 27B를 사용(단일 모델 다역할 원칙 유지).

전략:
  - rewrite   : 원 질의를 검색 친화적으로 재작성 (대명사 해소, 핵심어 강조)
  - decompose : 복합 질의를 1~3개 하위 질의로 분해 (multi-hop 대응)
  - hyde      : 가상의 이상적 답변을 생성해 그 임베딩으로 검색 (Hypothetical Document Embeddings)

graph.py의 retrieve가 retry 단계에 따라 어떤 전략을 쓸지는
adaptive_retrieve(아래 별도 파일)에서 결정한다.
"""

from __future__ import annotations

from harness.prompts import load_prompt
from models.backends import ask

# 쿼리 확장은 약간의 다양성 허용(temperature=0.3). 모델/엔드포인트는 backends가 관리.


def _ask(system: str, user: str) -> str:
    return ask(system, user, temperature=0.3, role="expand")


# 시스템 프롬프트는 prompts/*.md 에서 로드 (코드 분리, 버전관리 용이)
REWRITE_PROMPT = load_prompt("rewrite")
DECOMPOSE_PROMPT = load_prompt("decompose")
HYDE_PROMPT = load_prompt("hyde")
GUIDED_REWRITE_PROMPT = load_prompt("guided_rewrite")


def rewrite(query: str) -> str:
    """검색 친화적 단일 재작성."""
    return _ask(REWRITE_PROMPT, query)


def guided_rewrite(query: str, diagnosis: str) -> str:
    """
    VERIFY 진단을 받아 '빠진 정보'를 겨냥해 재작성.
    diagnosis가 비어있으면 일반 rewrite로 폴백.
    """
    if not diagnosis or not diagnosis.strip():
        return rewrite(query)
    user = f"[원 질문]\n{query}\n\n[부족한 점]\n{diagnosis}"
    out = _ask(GUIDED_REWRITE_PROMPT, user)
    return out if 0 < len(out) <= 300 else query


def decompose(query: str) -> list[str]:
    """복합 질의 → 하위 질의 리스트 (1~3개)."""
    raw = _ask(DECOMPOSE_PROMPT, query)
    subs = [ln.strip(" -•\t") for ln in raw.splitlines() if ln.strip()]
    return subs[:3] if subs else [query]


def hyde(query: str) -> str:
    """
    HyDE: 가상 답변을 반환. 호출부에서 이 텍스트를 임베딩해
    질의 벡터 대신 사용한다(원 질의보다 정답 문서에 가까운 벡터를 얻기 위함).
    """
    return _ask(HYDE_PROMPT, query)

"""
gemma-rag-harness / models/backends.py

모델 백엔드 추상화. 역할(role)별로 다른 모델 티어를 쓸 수 있다.

기본 구성: Gemma 4 E4B + MTP(추측 디코딩) 단일 모델.
  - E4B(8B)에 MTP 초안 모델을 붙여 vLLM 하나로 서빙(포트 8000).
  - MTP로 생성이 빨라 ROUTER/VERIFY/GENERATE를 모두 같은 모델로 처리해도
    챗봇 수준 응답 속도가 나온다. 별도 작은 모델 불필요.

역할 분리가 필요하면(예: GENERATE만 더 큰 모델) 환경변수로 티어를 나눌 수 있다:
  HEAVY_MODEL / HEAVY_URL   기본 google/gemma-4-E4B-it / http://localhost:8000/v1
  LIGHT_MODEL / LIGHT_URL   기본 heavy와 동일(단일). 분리 시 다른 포트 지정.
  LLM_API_KEY               로컬 vLLM은 EMPTY (기본)

  # 하위호환: LLM_MODEL / VLLM_URL 을 주면 heavy 기본값을 덮어쓴다.
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_openai import ChatOpenAI

# .env 자동 로딩 (있으면). 없거나 dotenv 미설치여도 조용히 진행.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")

# ── 두 티어 정의 ─────────────────────────────────────────
# 하위호환: 예전 변수(LLM_MODEL/VLLM_URL)가 있으면 heavy 기본값으로 사용.
HEAVY_MODEL = os.getenv("HEAVY_MODEL", os.getenv("LLM_MODEL", "google/gemma-4-E4B-it"))
HEAVY_URL = os.getenv("HEAVY_URL", os.getenv("VLLM_URL", "http://localhost:8000/v1"))

# 라이트 티어: 기본은 heavy와 동일(단일 모델). MTP로 E4B가 충분히 빨라
# 별도 작은 모델이 불필요하다. 원하면 LIGHT_URL/LIGHT_MODEL로 분리 가능.
LIGHT_MODEL = os.getenv("LIGHT_MODEL", HEAVY_MODEL)
LIGHT_URL = os.getenv("LIGHT_URL", HEAVY_URL)

# 라이트 티어 사용 여부. LIGHT_URL 이 heavy 와 같으면(=따로 안 띄움) 자동 비활성.
_LIGHT_ENABLED = LIGHT_URL != HEAVY_URL

# 역할 → 티어 매핑. 여기만 바꾸면 어떤 역할을 어느 모델로 보낼지 조정된다.
_ROLE_TIER = {
    "router": "light",     # 질의 분류: 단순 → 작은 모델
    "verify": "light",     # 근거 충실성 판정: 단순 → 작은 모델
    "generate": "heavy",   # 답변 생성: 품질 → 27B
    "direct": "heavy",     # 검색 없이 직접 답변: 품질 → 27B
    "expand": "light",     # 쿼리 재작성/분해/HyDE: 단순 → 작은 모델
    "judge": "heavy",      # 평가(LLM-as-Judge): 채점 품질 → 27B
}

_TIERS = {
    "heavy": (HEAVY_MODEL, HEAVY_URL),
    "light": (LIGHT_MODEL, LIGHT_URL) if _LIGHT_ENABLED else (HEAVY_MODEL, HEAVY_URL),
}


def _resolve(role: str | None) -> tuple[str, str]:
    """역할 이름 → (모델ID, 엔드포인트). 미지정/미등록이면 heavy."""
    tier = _ROLE_TIER.get(role or "", "heavy")
    return _TIERS[tier]


@lru_cache(maxsize=32)
def get_llm(temperature: float = 0.0, timeout: int = 60,
            max_tokens: int | None = None,
            role: str | None = None) -> ChatOpenAI:
    """
    역할에 맞는 LLM 클라이언트를 반환(파라미터별 캐시).
    role 로 heavy/light 티어를 선택한다. max_tokens=None이면 서버 기본값.
    """
    model_id, base_url = _resolve(role)
    return ChatOpenAI(
        model=model_id,
        base_url=base_url,
        api_key=_API_KEY,           # 로컬 서버는 키 불필요
        temperature=temperature,
        timeout=timeout,
        max_tokens=max_tokens,
    )


def ask(system: str, user: str, temperature: float = 0.0,
        max_tokens: int | None = None, role: str | None = None) -> str:
    """
    단일 system+user 호출. 하네스 노드들이 공통으로 쓴다.
      role   : "router"/"verify"/"generate"/"direct"/"expand" 중 하나.
               기본 구성(단일 모델)에선 모두 같은 E4B로 간다. 티어를 분리한
               경우 라이트 역할은 LIGHT_URL, 헤비 역할은 HEAVY_URL로 간다.
      max_tokens : 답변 길이 제한. GENERATE·VERIFY 지연을 크게 줄인다.
    """
    llm = get_llm(temperature=temperature, max_tokens=max_tokens, role=role)
    return llm.invoke([("system", system), ("user", user)]).content.strip()


if __name__ == "__main__":
    print(f"[heavy] {HEAVY_MODEL} @ {HEAVY_URL}")
    print(f"[light] {LIGHT_MODEL} @ {LIGHT_URL}  (enabled={_LIGHT_ENABLED})")
    for r in ("router", "verify", "generate", "direct", "expand"):
        m, u = _resolve(r)
        print(f"  role={r:9s} → {m} @ {u}")

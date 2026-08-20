"""OpenAI 호환 LLM 클라이언트.

하네스와 동일한 추상화: base_url + model 두 값만 바꾸면
vLLM(워크스테이션, Gemma 4 E4B+MTP) <-> Ollama(집 노트북) 전환.
카드 프론트매터 `model:` 필드(작성 모델 출처 기록)에 model_name을 그대로 쓴다.
"""

from __future__ import annotations

import json
import re
from typing import Any

try:
    import httpx as _http
    _HTTP = "httpx"
except ImportError:  # pragma: no cover
    try:
        import requests as _http  # type: ignore
        _HTTP = "requests"
    except ImportError as e:
        raise ImportError(
            "httpx 또는 requests가 필요합니다: pip install httpx") from e

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


class LLMClient:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    # ---------- 기본 채팅 ----------
    def chat(self, system: str, user: str, temperature: float = 0.2,
             max_tokens: int = 2048) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        url = f"{self.base_url}/chat/completions"
        if _HTTP == "httpx":
            r = _http.post(url, json=payload, timeout=self.timeout)
        else:
            r = _http.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()

    # ---------- JSON 응답 ----------
    def chat_json(self, system: str, user: str,
                  retries: int = 1) -> dict | list:
        sys_p = (system + "\n\n반드시 유효한 JSON만 출력한다. "
                 "마크다운 펜스, 서두, 해설 금지.")
        last_err: Exception | None = None
        for _ in range(retries + 1):
            raw = self.chat(sys_p, user, temperature=0.1)
            try:
                return _parse_json(raw)
            except ValueError as e:
                last_err = e
                user = (f"{user}\n\n이전 출력이 JSON 파싱에 실패했다: {e}. "
                        f"순수 JSON만 다시 출력하라.")
        raise ValueError(f"LLM JSON 파싱 실패: {last_err}")


def _parse_json(raw: str) -> dict | list:
    m = _FENCE_RE.search(raw)
    if m:
        raw = m.group(1)
    raw = raw.strip()
    start = min([i for i in (raw.find("{"), raw.find("[")) if i >= 0],
                default=-1)
    if start > 0:
        raw = raw[start:]
    return json.loads(raw)

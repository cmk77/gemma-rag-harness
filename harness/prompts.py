"""
gemma-rag-harness / harness/prompts.py

prompts/*.md 시스템 프롬프트 로더.
프롬프트를 코드에서 분리해(ADR-0003 관련 원칙) 버전 관리·수정을 쉽게 한다.
프롬프트 개선이 코드 diff가 아니라 prompts/ 파일 diff로 추적된다.

사용:
  from harness.prompts import load_prompt
  ROUTER_PROMPT = load_prompt("router")
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# 저장소 루트 기준 prompts/ (이 파일: harness/prompts.py → 루트는 부모의 부모)
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    """
    prompts/<name>.md 를 읽어 반환(캐시). 끝의 개행은 정리.
    누락 시 사용 가능한 프롬프트 목록과 함께 명확히 실패.
    """
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        available = sorted(p.stem for p in _PROMPTS_DIR.glob("*.md"))
        raise FileNotFoundError(
            f"프롬프트 '{name}' 없음: {path}\n사용 가능: {available}"
        )
    return path.read_text(encoding="utf-8").strip()


def available_prompts() -> list[str]:
    """등록된 프롬프트 이름 목록."""
    return sorted(p.stem for p in _PROMPTS_DIR.glob("*.md"))


if __name__ == "__main__":
    for name in available_prompts():
        text = load_prompt(name)
        print(f"[{name}] {len(text)}자")
        print(f"  {text.splitlines()[0]}")

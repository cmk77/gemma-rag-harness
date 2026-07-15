"""
tests/test_prompts.py

프롬프트 로더와 회귀 감지 로직 테스트. 외부 의존 없음.
"""

from __future__ import annotations

import pytest

from harness.prompts import available_prompts, load_prompt


# ── 프롬프트 로더 ────────────────────────────────────────
def test_all_required_prompts_exist():
    """하네스가 쓰는 7개 프롬프트가 모두 존재해야."""
    names = set(available_prompts())
    required = {"router", "generate", "verify", "rewrite",
               "guided_rewrite", "decompose", "hyde"}
    assert required <= names, f"누락: {required - names}"


def test_load_prompt_content():
    """로드한 프롬프트가 기대 키워드를 포함."""
    assert "retrieve" in load_prompt("router")
    assert "근거" in load_prompt("generate")
    assert "insufficient" in load_prompt("verify")


def test_load_prompt_stripped():
    """끝 공백/개행이 정리되어 있어야."""
    p = load_prompt("router")
    assert p == p.strip()


def test_missing_prompt_raises():
    """없는 프롬프트는 명확한 에러 + 사용가능 목록."""
    with pytest.raises(FileNotFoundError) as exc:
        load_prompt("does_not_exist")
    assert "사용 가능" in str(exc.value)


def test_load_prompt_cached():
    """같은 이름은 캐시되어 동일 객체 반환(lru_cache)."""
    assert load_prompt("router") is load_prompt("router")


# ── 회귀 감지 ────────────────────────────────────────────
def test_regression_detects_recall_drop():
    from eval.run_regression import detect_regression
    base = {"retrieval": {"Recall@5": 0.9, "MRR": 0.85}, "judge": {}}
    cur = {"retrieval": {"Recall@5": 0.7, "MRR": 0.85}, "judge": {}}
    failures = detect_regression(cur, base)
    assert len(failures) == 1 and "Recall@5" in failures[0]


def test_regression_ignores_improvement():
    from eval.run_regression import detect_regression
    base = {"retrieval": {"Recall@5": 0.8}, "judge": {}}
    cur = {"retrieval": {"Recall@5": 0.95}, "judge": {}}
    assert detect_regression(cur, base) == []


def test_regression_ignores_noise():
    """임계 이내 미세 하락은 회귀 아님."""
    from eval.run_regression import detect_regression
    base = {"retrieval": {"Recall@5": 0.90}, "judge": {}}
    cur = {"retrieval": {"Recall@5": 0.88}, "judge": {}}  # 0.02 하락 < 0.05 임계
    assert detect_regression(cur, base) == []


def test_regression_detects_judge_drop():
    from eval.run_regression import detect_regression
    base = {"retrieval": {}, "judge": {"faithfulness": 4.5, "correctness": 4.2}}
    cur = {"retrieval": {}, "judge": {"faithfulness": 4.0, "correctness": 3.5}}
    failures = detect_regression(cur, base)
    assert len(failures) == 2


def test_regression_skips_missing_metrics():
    """한쪽에 없는 지표는 비교 생략(크래시 없음)."""
    from eval.run_regression import detect_regression
    base = {"retrieval": {"Recall@5": 0.9}, "judge": {}}
    cur = {"retrieval": {}, "judge": {}}  # Recall 없음
    assert detect_regression(cur, base) == []

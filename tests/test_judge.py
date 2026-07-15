"""
tests/test_judge.py

LLM-as-Judge의 파싱·집계 견고성 테스트.
모델 호출 없이 파서 로직만 검증(langchain 스텁만 필요).
핵심: 측정 실패(파싱 깨짐)를 0점이 아닌 None으로 격리하는지.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _stub(langchain_stub):
    """judge 모듈 import 전에 langchain 스텁 설치(응답은 안 씀, import만 통과)."""
    langchain_stub(lambda system, user: "")


# ── JSON 추출 ────────────────────────────────────────────
def test_extract_pure_json():
    from eval.judge import _extract_json
    r = _extract_json('{"faithfulness": 5, "correctness": 4, "relevance": 5, "reason": "x"}')
    assert r["faithfulness"] == 5


def test_extract_code_fenced():
    """```json 코드펜스를 벗겨내야."""
    from eval.judge import _extract_json
    r = _extract_json('```json\n{"faithfulness": 3, "relevance": 4}\n```')
    assert r["faithfulness"] == 3


def test_extract_with_surrounding_text():
    """앞뒤 잡텍스트가 있어도 JSON만 추출."""
    from eval.judge import _extract_json
    r = _extract_json('채점 결과: {"correctness": 4} 입니다.')
    assert r["correctness"] == 4


def test_extract_malformed_returns_none():
    """깨진 형식은 None(측정 실패 신호)."""
    from eval.judge import _extract_json
    assert _extract_json("판정 불가") is None
    assert _extract_json("{불완전") is None


# ── 점수 클램프 ──────────────────────────────────────────
def test_clamp_in_range():
    from eval.judge import _clamp
    assert _clamp(3) == 3
    assert _clamp("4") == 4
    assert _clamp(3.6) == 4  # 반올림


def test_clamp_out_of_range():
    """1~5 범위로 강제."""
    from eval.judge import _clamp
    assert _clamp(7) == 5
    assert _clamp(0) == 1
    assert _clamp(-5) == 1


def test_clamp_invalid_returns_none():
    from eval.judge import _clamp
    assert _clamp(None) is None
    assert _clamp("bad") is None


# ── 집계: 측정 실패 격리 (핵심) ───────────────────────────
def test_aggregate_excludes_parse_failures():
    """파싱 실패건은 평균에서 제외 — 0점 오염 금지."""
    from eval.judge import aggregate_judgments
    judgments = [
        {"faithfulness": 5, "correctness": 4, "relevance": 5, "parsed": True},
        {"faithfulness": 3, "correctness": 2, "relevance": 4, "parsed": True},
        {"faithfulness": None, "correctness": None, "relevance": None, "parsed": False},
    ]
    agg = aggregate_judgments(judgments)
    assert agg["n_total"] == 3
    assert agg["n_parsed"] == 2
    # (5+3)/2 = 4.0 — 실패건의 None이 0으로 섞이지 않음
    assert agg["faithfulness"] == 4.0
    assert agg["correctness"] == 3.0


def test_aggregate_parse_fail_rate():
    from eval.judge import aggregate_judgments
    judgments = [
        {"faithfulness": 5, "correctness": 5, "relevance": 5, "parsed": True},
        {"parsed": False},
        {"parsed": False},
    ]
    agg = aggregate_judgments(judgments)
    assert abs(agg["parse_fail_rate"] - 2 / 3) < 1e-4


def test_aggregate_all_failed():
    """전부 실패해도 크래시 없이 None 반환."""
    from eval.judge import aggregate_judgments
    agg = aggregate_judgments([{"parsed": False}, {"parsed": False}])
    assert agg["faithfulness"] is None
    assert agg["parse_fail_rate"] == 1.0

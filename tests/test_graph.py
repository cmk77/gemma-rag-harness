"""
tests/test_graph.py

하네스 그래프의 제어 흐름 통합 테스트.
스텁 LLM/ES로 모델 없이 검증한다 — 이 프로젝트의 핵심 가치인
'진단 기반 재검색 루프'와 '루프 가드'가 실제로 동작하는지.
"""

from __future__ import annotations

import pytest


def _default_responder(system: str, user: str) -> str:
    """시스템 프롬프트로 역할을 식별해 응답 결정."""
    if "질의 라우터" in system:
        return "retrieve"
    if "재작성기" in system:
        return "재작성된 검색 질의"
    if "분해기" in system:
        return "하위질의1\n하위질의2"
    if "가상 답변" in system:
        return "가상 답변 단락"
    if "검증기" in system:
        # 근거 섹션에 'rank_constant'가 있으면 ok, 없으면 insufficient.
        # (질문에도 rank_constant가 들어올 수 있으므로 [근거] 이후만 검사)
        evidence = user.split("[근거]", 1)[-1]
        return "ok\n" if "rank_constant" in evidence else "insufficient\nrank_constant 근거 부족"
    if "답변 생성기" in system:
        return "생성된 답변 초안"
    return "기타"


@pytest.fixture
def make_app(langchain_stub, langgraph_stub, retrieval_stub):
    """스텁을 설치하고 build_graph()를 반환하는 팩토리."""
    def _make(responder=_default_responder, search_behavior=None):
        langchain_stub(responder)
        if search_behavior is None:
            from retrieval.vector_store import Doc
            # 기본: 1차는 무관 문서, 2차부터 정답(rank_constant 포함)
            def search_behavior(call_n, query):
                if call_n == 1:
                    return [Doc("irr", "무관한 문서", 0.5)]
                return [Doc("ans", "rank_constant는 상위 영향력을 조절한다", 0.95)]
        retrieval_stub(search_behavior)
        # 스텁 설치 후 의존 모듈을 모두 reload해야 함.
        # models.backends는 lru_cache로 LLM 인스턴스를 캐시하므로, 새 스텁을
        # 잡으려면 backends를 reload하고 캐시를 비워야 한다.
        # 순서: backends → query_expansion → adaptive_retrieve → graph
        import importlib

        from harness import adaptive_retrieve, graph, query_expansion
        from models import backends
        importlib.reload(backends)
        backends.get_llm.cache_clear()
        importlib.reload(query_expansion)
        importlib.reload(adaptive_retrieve)
        importlib.reload(graph)
        return graph.build_graph()
    return _make


# ── 재검색 루프 ──────────────────────────────────────────
def test_requery_loop_recovers(make_app):
    """1차 검색 실패 → 진단 기반 재검색 → 정답 확보 → ok 종료."""
    app = make_app()
    result = app.invoke({"query": "rank_constant의 역할은?"})
    trace = result["_trace"]
    # retrieve가 2회 이상(재검색 발생)
    assert trace.count("retrieve") >= 2
    assert result["verdict"] == "ok"
    assert result.get("stop_reason") == "verified"
    assert "answer" in result


def test_diagnosis_passed_to_requery(make_app):
    """VERIFY 진단이 재검색에 전달되는지(diagnosis 상태에 실림)."""
    captured = {}

    def responder(system, user):
        if "재작성기" in system and "부족한 점" in user:
            captured["diagnosis_seen"] = True
        return _default_responder(system, user)

    app = make_app(responder=responder)
    app.invoke({"query": "rank_constant의 역할은?"})
    # guided_rewrite가 진단을 받아 호출되었는지
    assert captured.get("diagnosis_seen") is True


# ── 루프 가드 ────────────────────────────────────────────
def test_loop_guard_stops_when_no_new_docs(make_app):
    """재검색해도 같은 문서만 나오면 상한 전 조기 종료."""
    from retrieval.vector_store import Doc

    def always_same(call_n, query):
        return [Doc("same", "항상 같은 무관 문서", 0.5)]  # rank_constant 없음 → 계속 insufficient

    app = make_app(search_behavior=always_same)
    result = app.invoke({"query": "답 없는 질문"})
    assert result.get("stop_reason") == "no_new_docs"
    assert result.get("no_new_docs") is True


def test_max_retries_respected(make_app):
    """새 문서는 계속 들어오지만 검증은 계속 실패 → 상한에서 종료."""
    from retrieval.vector_store import Doc

    def always_new_but_irrelevant(call_n, query):
        # 매번 새 id지만 rank_constant 없음 → 계속 insufficient
        return [Doc(f"new{call_n}", f"무관 문서 {call_n}", 0.5)]

    app = make_app(search_behavior=always_new_but_irrelevant)
    result = app.invoke({"query": "답 없는 질문"})
    assert result.get("stop_reason") == "max_retries"
    # retrieve 호출이 무한이 아니라 상한 + 1로 제한됨
    assert result["_trace"].count("retrieve") <= 3


# ── 라우팅 분기 ──────────────────────────────────────────
def test_route_direct(make_app):
    """direct로 분류되면 검색 없이 바로 답변."""
    def responder(system, user):
        if "질의 라우터" in system:
            return "direct"
        if "친절한" in system:
            return "직접 답변"
        return _default_responder(system, user)

    app = make_app(responder=responder)
    result = app.invoke({"query": "안녕"})
    assert "retrieve" not in result["_trace"]
    assert result["_trace"] == ["router", "direct"]


def test_route_reject(make_app):
    """reject로 분류되면 거절 메시지."""
    def responder(system, user):
        if "질의 라우터" in system:
            return "reject"
        return _default_responder(system, user)

    app = make_app(responder=responder)
    result = app.invoke({"query": "범위 밖 질문"})
    assert result["_trace"] == ["router", "reject"]
    assert "범위" in result["answer"]


def test_router_fallback_to_retrieve(make_app):
    """라우터가 이상한 값을 내면 보수적으로 retrieve."""
    def responder(system, user):
        if "질의 라우터" in system:
            return "garbage_value"  # 유효하지 않은 라우팅
        return _default_responder(system, user)

    app = make_app(responder=responder)
    result = app.invoke({"query": "rank_constant?"})
    assert "retrieve" in result["_trace"]

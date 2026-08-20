"""LangGraph 통합 레이어.

기존 하네스(ROUTER -> RETRIEVE -> GENERATE -> VERIFY + 재검색 루프)와
동일한 골격의 위키 서브그래프를 제공한다. 두 가지 사용 방식:

  A) 서브그래프 마운트: build_wiki_graph(engine)로 컴파일된 그래프를
     하네스 ROUTER의 'wiki' 분기 대상으로 add_node.
  B) 도구 함수: wiki_query_tool(engine) -> Callable[str, dict]
     ROUTER가 함수 호출로 쓰는 가장 단순한 형태.

langgraph 미설치 환경에서도 QueryEngine 단독 사용이 가능하도록
import는 함수 내부로 지연시킨다.
"""

from __future__ import annotations

from typing import Any, Callable, TypedDict

from .query import QueryEngine


class WikiState(TypedDict, total=False):
    question: str
    wide: bool
    cards: list
    answer: str
    verified: bool
    retries: int
    trace: list[str]


def build_wiki_graph(engine: QueryEngine):
    """retrieve -> generate_verify -> (통과: END | 실패: retrieve[wide]) 루프."""
    from langgraph.graph import END, StateGraph

    def retrieve(state: WikiState) -> WikiState:
        from .query import QueryTrace
        trace = QueryTrace()
        cards = engine._retrieve(  # noqa: SLF001 — 엔진 내부 단계 재사용
            state["question"], wide=state.get("wide", False), trace=trace)
        return {"cards": cards,
                "trace": state.get("trace", []) + trace.steps}

    def generate_verify(state: WikiState) -> WikiState:
        from .query import QueryTrace
        trace = QueryTrace()
        answer, ok = engine._generate_and_verify(  # noqa: SLF001
            state["question"], state.get("cards", []), trace)
        return {"answer": answer, "verified": ok,
                "retries": state.get("retries", 0),
                "trace": state.get("trace", []) + trace.steps}

    def widen(state: WikiState) -> WikiState:
        return {"wide": True, "retries": state.get("retries", 0) + 1}

    def route_after_verify(state: WikiState) -> str:
        if state.get("verified"):
            return "done"
        if state.get("retries", 0) >= engine.s.max_retries:
            return "done"
        return "retry"

    g = StateGraph(WikiState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate_verify", generate_verify)
    g.add_node("widen", widen)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate_verify")
    g.add_conditional_edges("generate_verify", route_after_verify,
                            {"done": END, "retry": "widen"})
    g.add_edge("widen", "retrieve")
    return g.compile()


def wiki_query_tool(engine: QueryEngine) -> Callable[[str], dict[str, Any]]:
    """하네스 ROUTER 분기용 최소 인터페이스.

    사용 예 (harness/graph.py ROUTER 노드):
        from wiki_vault.bootstrap import build_default_engine
        from wiki_vault.graph import wiki_query_tool
        wiki_q = wiki_query_tool(build_default_engine())
        ...
        if route == "wiki":
            return wiki_q(state["question"])
    """
    def _tool(question: str) -> dict[str, Any]:
        r = engine.run(question)
        return {"answer": r.answer, "sources": r.sources,
                "verified": r.verified, "retries": r.retries,
                "trace": r.trace.steps, "route": "wiki"}
    return _tool

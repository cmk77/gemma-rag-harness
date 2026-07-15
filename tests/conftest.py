"""
tests/conftest.py

모델·ES 없이 테스트가 돌도록 외부 의존을 가짜로 주입한다.
CLAUDE.md 규약: tests/는 GPU·ES 없이 통과해야 한다.

스텁 대상:
  - langchain_openai.ChatOpenAI : 시스템 프롬프트로 역할을 식별해 정해진 응답 반환
  - langgraph.graph.StateGraph  : 조건부 엣지/순환을 지원하는 최소 실행기
  - retrieval.es_store / indexer: 호출 순서에 따라 다른 문서를 주는 가짜 검색
"""

from __future__ import annotations

import sys
import types

import pytest


def _install_langchain_stub(responder):
    """responder(system, user) -> str 로 응답을 결정하는 ChatOpenAI 스텁 설치."""
    lc = types.ModuleType("langchain_openai")

    class _Resp:
        def __init__(self, c):
            self.content = c

    class ChatOpenAI:
        def __init__(self, **kw):
            self.kw = kw

        def invoke(self, msgs):
            system = msgs[0][1]
            user = msgs[1][1]
            return _Resp(responder(system, user))

    lc.ChatOpenAI = ChatOpenAI
    sys.modules["langchain_openai"] = lc


def _install_langgraph_stub():
    """조건부 엣지 + 순환을 지원하는 StateGraph 최소 구현."""
    lg = types.ModuleType("langgraph.graph")
    END = "__END__"

    class StateGraph:
        def __init__(self, schema):
            self.nodes = {}
            self.edges = {}
            self.cond = {}
            self.entry = None

        def add_node(self, n, fn):
            self.nodes[n] = fn

        def add_edge(self, a, b):
            self.edges[a] = b

        def add_conditional_edges(self, n, fn, mapping):
            self.cond[n] = (fn, mapping)

        def set_entry_point(self, n):
            self.entry = n

        def compile(self):
            return _Compiled(self)

    class _Compiled:
        def __init__(self, g):
            self.g = g

        def invoke(self, state, max_steps=30):
            g = self.g
            node = g.entry
            steps = 0
            trace = []
            while node != END and steps < max_steps:
                steps += 1
                trace.append(node)
                out = g.nodes[node](state) or {}
                state = {**state, **out}
                if node in g.cond:
                    fn, mapping = g.cond[node]
                    node = mapping[fn(state)]
                elif node in g.edges:
                    node = g.edges[node]
                else:
                    break
            state["_trace"] = trace
            return state

    lg.StateGraph = StateGraph
    lg.END = END
    sys.modules["langgraph.graph"] = lg
    sys.modules["langgraph"] = types.ModuleType("langgraph")


def _install_retrieval_stub(search_behavior):
    """retrieval.es_store.ESStore + indexer 스텁. search_behavior(call_n, query)->list[Doc]."""
    from retrieval.vector_store import Doc

    es = types.ModuleType("retrieval.es_store")
    counter = {"n": 0}

    class ESStore:
        def __init__(self, *a, **k):
            pass

        def hybrid_search(self, query, qvec, k=5):
            counter["n"] += 1
            return search_behavior(counter["n"], query)

    es.ESStore = ESStore
    es.Doc = Doc
    sys.modules["retrieval.es_store"] = es

    idx = types.ModuleType("retrieval.indexer")
    idx.embed_query = lambda q: [0.0] * 8
    idx.embed = lambda ts: [[0.0] * 8 for _ in ts]
    sys.modules["retrieval.indexer"] = idx
    return counter


# ── 공개 픽스처 ──────────────────────────────────────────
@pytest.fixture
def langchain_stub():
    """기본 responder를 주는 픽스처. 테스트가 responder를 커스텀할 수 있게 팩토리 반환."""
    return _install_langchain_stub


@pytest.fixture
def langgraph_stub():
    _install_langgraph_stub()


@pytest.fixture
def retrieval_stub():
    return _install_retrieval_stub

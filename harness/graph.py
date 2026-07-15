"""
gemma-rag-harness / harness/graph.py

Gemma 4 27B를 코어 오케스트레이터로 쓰는 RAG 에이전트 하네스 (LangGraph).
단일 모델을 ROUTER / GENERATE / VERIFY 세 역할로 다르게 프롬프트하여 호출.

S1 단계: RETRIEVE를 mock으로 두면 검색 없이도 그래프가 돌아간다.
가중치/ES 연결 시 retrieval.es_store 구현으로 교체하면 프로덕션 전환.

실행:
  # 1) Gemma 서빙 (vLLM, OpenAI 호환)
  #   vllm serve google/gemma-3-27b-it --port 8000
  # 2) pip install langgraph langchain-openai
  # 3) python -m harness.graph
"""

from __future__ import annotations

import logging
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from models.backends import ask as _ask

# 하네스 실행 경로 로거. scripts/ask.py 등에서 레벨을 INFO로 켜면
# 각 노드가 실행될 때마다 어떤 노드를 지나는지 보인다.
logger = logging.getLogger("harness")

# ─────────────────────────────────────────────────────────────
# 모델 호출은 models/backends.py의 ask()로 위임 (ADR-0003: 단일 모델 다역할).
#   LLM_BACKEND 환경변수로 vLLM/Ollama 전환, base_url·모델ID는 backends가 관리.
#   ROUTER/GENERATE/VERIFY 모두 같은 모델을 다른 시스템 프롬프트로 호출.

MAX_RETRIES = 2               # VERIFY → RETRIEVE 재진입 상한

# GENERATE 답변 생성 토큰 상한. 27B는 토큰 하나씩 생성하므로 이 값이
# 응답 시간을 좌우한다(작을수록 빠름). 간결한 답변용으로 256 설정.
GENERATE_MAX_TOKENS = 256

# VERIFY 판정 토큰 상한. 판정 한 줄 + 진단 한 줄이면 충분하므로 짧게 제한해
# 검증 단계 지연을 줄인다.
VERIFY_MAX_TOKENS = 128


# ─────────────────────────────────────────────────────────────
# 상태: 노드 간 전달되는 그래프 상태
# ─────────────────────────────────────────────────────────────
class AgentState(TypedDict, total=False):
    query: str                # 사용자 질의
    route: str                # router 분류 결과: retrieve / direct / reject
    docs: list[str]           # 검색된 근거 청크
    draft: str                # 생성된 답변 초안
    verdict: str              # 검증 결과: ok / insufficient
    diagnosis: str            # insufficient일 때 '무엇이 부족한지' (재검색 재작성 재료)
    retry_count: int          # 재검색 횟수
    no_new_docs: bool         # 재검색이 새 문서를 못 가져옴 (루프 가드)
    stop_reason: str          # 종료 사유: verified / max_retries / no_new_docs
    _doc_pool: list           # 재진입 간 누적되는 검색 문서 풀 (adaptive_retrieve 내부용)
    answer: str               # 최종 답변


# ─────────────────────────────────────────────────────────────
# 프롬프트 (실제 프로젝트에서는 prompts/*.md로 분리·버전관리)
# ─────────────────────────────────────────────────────────────
from harness.prompts import load_prompt

# 시스템 프롬프트는 prompts/*.md 에서 로드 (코드 분리, 버전관리 용이)
ROUTER_PROMPT = load_prompt("router")
GENERATE_PROMPT = load_prompt("generate")
VERIFY_PROMPT = load_prompt("verify")


# ─────────────────────────────────────────────────────────────
# 노드들
# ─────────────────────────────────────────────────────────────
def router(state: AgentState) -> AgentState:
    route = _ask(ROUTER_PROMPT, state["query"], role="router").lower()
    if route not in {"retrieve", "direct", "reject"}:
        route = "retrieve"          # 파싱 실패 시 보수적으로 검색
    logger.info("ROUTER   → 질의 분류: %s", route)
    return {"route": route, "retry_count": state.get("retry_count", 0)}


def retrieve(state: AgentState) -> AgentState:
    """
    적응형 재검색에 위임. retry 단계와 VERIFY 진단에 따라 검색 입력을 바꾸고,
    재진입 간 문서를 누적하며, 새 문서를 못 가져오면 루프 가드 신호를 올린다.
    S1처럼 ES 없이 돌려보려면 아래 import를 주석 처리하고 mock 블록을 쓰면 된다.
    """
    from harness.adaptive_retrieve import adaptive_retrieve
    retry = state.get("retry_count", 0)
    result = adaptive_retrieve(dict(state))
    n_docs = len(result.get("docs", []))
    if retry == 0:
        logger.info("RETRIEVE → 검색 수행, 근거 %d개 확보", n_docs)
    else:
        logger.info("RETRIEVE → 재검색 %d회차, 근거 %d개 (진단 기반 재작성)", retry, n_docs)
    return result

    # ── S1 mock (ES 미연결 시) ─────────────────────────────
    # docs = [
    #     "Elasticsearch 8.x는 rank: rrf 로 BM25와 kNN 결과의 RRF 융합을 지원한다.",
    #     "RRF는 각 결과의 순위 역수를 합산해 점수화하며 스케일 보정이 필요 없다.",
    # ]
    # return {"docs": docs, "retry_count": state.get("retry_count", 0)}


def generate(state: AgentState) -> AgentState:
    ctx = "\n".join(f"[{i+1}] {d}" for i, d in enumerate(state.get("docs", [])))
    user = f"[근거]\n{ctx}\n\n[질문]\n{state['query']}"
    logger.info("GENERATE → 근거 기반 답변 생성 중...")
    return {"draft": _ask(GENERATE_PROMPT, user, max_tokens=GENERATE_MAX_TOKENS, role="generate")}


def verify(state: AgentState) -> AgentState:
    ctx = "\n".join(f"[{i+1}] {d}" for i, d in enumerate(state.get("docs", [])))
    # 질문도 넘겨야 VERIFY가 "답변이 질문에 맞는가"를 판정할 수 있다.
    user = f"[질문]\n{state['query']}\n\n[근거]\n{ctx}\n\n[답변]\n{state['draft']}"
    raw = _ask(VERIFY_PROMPT, user, max_tokens=VERIFY_MAX_TOKENS, role="verify")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    verdict = "ok" if (lines and "ok" in lines[0].lower()) else "insufficient"
    out: AgentState = {"verdict": verdict}
    if verdict == "ok":
        logger.info("VERIFY   → 근거 충실성 판정: ok (검증 통과)")
        out["answer"] = state["draft"]
        out["stop_reason"] = "verified"          # 노드에서 설정해야 최종 상태에 반영됨
    else:
        # 둘째 줄 진단을 다음 재검색의 쿼리 재작성 재료로 넘긴다
        diagnosis = lines[1] if len(lines) > 1 else ""
        new_retry = state.get("retry_count", 0) + 1
        out["diagnosis"] = diagnosis
        out["retry_count"] = new_retry
        # 이번 판정으로 루프가 끝날지(상한/루프가드) 여기서 미리 사유를 정한다.
        if state.get("no_new_docs", False):
            out["stop_reason"] = "no_new_docs"
            logger.info("VERIFY   → insufficient + 새 문서 없음 → 재검색 중단(no_new_docs)")
        elif new_retry >= MAX_RETRIES:
            out["stop_reason"] = "max_retries"
            logger.info("VERIFY   → insufficient, 재검색 상한(%d회) 도달 → 종료", MAX_RETRIES)
        else:
            logger.info("VERIFY   → 판정: insufficient (근거 부족) → 재검색으로")
    return out


def direct(state: AgentState) -> AgentState:
    logger.info("DIRECT   → 검색 없이 직접 답변")
    return {"answer": _ask("너는 친절한 어시스턴트다. 간결히 답하라.", state["query"], role="direct")}


def reject(state: AgentState) -> AgentState:
    logger.info("REJECT   → 범위 밖 질의, 거절")
    return {"answer": "죄송하지만 이 질문은 제가 다루는 지식 범위를 벗어납니다."}


# ─────────────────────────────────────────────────────────────
# 조건부 엣지
# ─────────────────────────────────────────────────────────────
def route_decision(state: AgentState) -> Literal["retrieve", "direct", "reject"]:
    return state["route"]            # type: ignore[return-value]


def verify_decision(state: AgentState) -> Literal["retry", "end"]:
    # 순수 판정만 한다. stop_reason 설정은 verify 노드에서 이미 끝났다
    # (조건부 엣지에서의 state 변경은 최종 상태에 반영되지 않으므로).
    if state["verdict"] == "ok":
        return "end"
    if state.get("no_new_docs", False):
        return "end"                 # 루프 가드
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "end"                 # 상한 도달
    return "retry"
    return "retry"


# ─────────────────────────────────────────────────────────────
# 그래프 조립
# ─────────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("router", router)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("verify", verify)
    g.add_node("direct", direct)
    g.add_node("reject", reject)

    g.set_entry_point("router")
    g.add_conditional_edges("router", route_decision, {
        "retrieve": "retrieve",
        "direct": "direct",
        "reject": "reject",
    })
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", verify_decision, {
        "retry": "retrieve",          # 재검색 루프
        "end": END,
    })
    g.add_edge("direct", END)
    g.add_edge("reject", END)

    return g.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({"query": "Elasticsearch에서 하이브리드 검색을 어떻게 융합하나요?"})
    print("\n=== 최종 답변 ===")
    print(result.get("answer"))
    print("\n=== 추적 ===")
    print({k: v for k, v in result.items() if k != "answer"})

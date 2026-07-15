"""
web/streamlit_app.py — Streamlit 챗봇 UI (전체 페이지형).

실행 (프로젝트 루트에서):
    pip install -e ".[web]"
    streamlit run web/streamlit_app.py
브라우저: http://localhost:8501

전제: vLLM(8000) + Elasticsearch(9200) + 색인 완료 (SETUP.md 4~5단계).

참고: Streamlit은 페이지 전체가 앱이라, "다른 웹페이지 우측 하단에 뜨는
플로팅 위젯" 형태로는 만들 수 없다. 그 UX가 필요하면 FastAPI 버전
(web/api.py + web/static/index.html)을 쓴다. 이 파일은 빠른 데모·내부
테스트용 챗 UI다.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# streamlit run은 스크립트 디렉토리를 sys.path에 넣으므로,
# 프로젝트 루트(harness/ 등)를 명시적으로 추가한다.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from web.capture import capture_path  # noqa: E402

STOP_KR = {
    "verified": "검증 통과",
    "max_retries": "재검색 상한 도달",
    "no_new_docs": "새 근거 없어 중단",
}

st.set_page_config(page_title="ExampleCorp 지식 챗봇", page_icon="💬", layout="centered")


@st.cache_resource(show_spinner="하네스 초기화 중… (그래프 + 임베딩 모델 로딩)")
def load_graph():
    """그래프 조립 + bge-m3 워밍업. cache_resource라 세션 간 1회만."""
    from harness.graph import build_graph
    graph = build_graph()
    try:
        from retrieval.indexer import embed_query
        embed_query("워밍업")
    except Exception:
        pass  # ES/모델 준비 전이어도 UI는 뜨게
    return graph


st.title("💬 ExampleCorp 지식 챗봇")
st.caption("Gemma 4 E4B + MTP · 자기검증 RAG (LangGraph + Elasticsearch 하이브리드 검색)")

with st.sidebar:
    st.markdown("### 사용 안내")
    st.markdown(
        "- **vLLM(8000)**·**ES(9200)**·색인이 준비돼 있어야 합니다.\n"
        "- 첫 실행은 임베딩 모델 로딩으로 느릴 수 있습니다.\n"
        "- 답변 아래에서 **근거**와 **실행 경로**를 펼쳐볼 수 있습니다."
    )
    st.markdown("### 예시 질문")
    st.markdown(
        "- ExampleCorp은 어떤 회사인가요?\n"
        "- GATEWAY-A이 받은 인증은?\n"
        "- PRODUCT-A로 API 만들 때 코딩이 필요한가요?"
    )
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_assistant(msg: dict) -> None:
    st.markdown(msg["content"])
    if msg.get("meta"):
        st.caption(msg["meta"])
    if msg.get("sources"):
        with st.expander(f"근거 {len(msg['sources'])}개"):
            for s in msg["sources"]:
                st.markdown(f"**[{s['index']}]** {s['preview']}…")
    if msg.get("path"):
        with st.expander("실행 경로"):
            st.code("\n".join(msg["path"]), language=None)


# 지난 대화 렌더
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            render_assistant(m)
        else:
            st.markdown(m["content"])

# 입력 처리
if q := st.chat_input("ExampleCorp에 대해 물어보세요…"):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("검색 → 생성 → 검증 중…"):
            t0 = time.time()
            try:
                graph = load_graph()
                with capture_path() as path:
                    result = graph.invoke({"query": q})

                docs = result.get("docs", [])
                stop = result.get("stop_reason", "n/a")
                msg = {
                    "role": "assistant",
                    "content": result.get("answer", "(답변 없음)"),
                    "meta": (
                        f"{STOP_KR.get(stop, stop)} · "
                        f"재검색 {result.get('retry_count', 0)}회 · "
                        f"{time.time() - t0:.1f}초"
                    ),
                    "sources": [
                        {"index": i + 1, "preview": d[:120].replace("\n", " ")}
                        for i, d in enumerate(docs[:5])
                    ],
                    "path": list(path),
                }
            except Exception as e:
                msg = {
                    "role": "assistant",
                    "content": (
                        "⚠ 서버에 연결할 수 없습니다. vLLM(8000)과 "
                        f"Elasticsearch(9200) 기동 여부를 확인하세요.\n\n`{str(e)[:200]}`"
                    ),
                }
        render_assistant(msg)
    st.session_state.messages.append(msg)

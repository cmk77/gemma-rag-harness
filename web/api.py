"""
web/api.py — FastAPI 챗봇 백엔드 + 플로팅 위젯 데모 페이지.

실행 (프로젝트 루트에서):
    pip install -e ".[web]"
    uvicorn web.api:app --host 0.0.0.0 --port 8080

브라우저에서 http://localhost:8080 접속 → 우측 하단 챗 버튼 클릭.

전제:
  - vLLM 서빙(포트 8000)과 Elasticsearch(9200)가 떠 있고 색인이 완료된 상태
    (SETUP.md 4~5단계). 안 떠 있으면 /api/chat이 503으로 안내를 돌려준다.
  - 포트 8080 사용: vLLM(8000)과 충돌하지 않도록.

동작:
  - 서버 기동 시 하네스 그래프를 1회 조립하고 bge-m3 임베딩을 워밍업해서
    첫 질문 지연(임베딩 모델 로딩 ~수십 초)을 제거한다.
  - 실행 경로(ROUTER→RETRIEVE→GENERATE→VERIFY)를 캡처해 응답에 포함한다.
  - 경로 캡처 특성상 질의는 직렬 처리된다(web/capture.py 참고) — 로컬 데모 기준.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from web.capture import capture_path

STATIC_DIR = Path(__file__).resolve().parent / "static"

STOP_KR = {
    "verified": "검증 통과",
    "max_retries": "재검색 상한 도달",
    "no_new_docs": "새 근거 없어 중단",
}

_graph = None


def get_graph():
    """하네스 그래프 싱글턴 — 프로세스에서 한 번만 조립."""
    global _graph
    if _graph is None:
        from harness.graph import build_graph
        _graph = build_graph()
    return _graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 워밍업: 그래프 조립 + bge-m3 로딩(첫 질문 지연 제거).
    # vLLM/ES가 아직 안 떠 있어도 서버 자체는 뜨도록 실패를 삼킨다.
    try:
        get_graph()
        from retrieval.indexer import embed_query
        embed_query("워밍업")
        print("[web] 워밍업 완료 (하네스 그래프 + bge-m3 임베딩)")
    except Exception as e:  # noqa: BLE001 — 데모 서버는 기동 우선
        print(f"[web] 워밍업 건너뜀 (첫 질문이 다소 느릴 수 있음): {e}")
    yield


app = FastAPI(title="gemma-rag-harness web", version="0.1.0", lifespan=lifespan)

# 위젯을 다른 사이트 페이지에 심어 쓰는 경우를 위해 CORS 개방(데모 기준).
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class ChatIn(BaseModel):
    question: str


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat")
def chat(body: ChatIn):
    q = body.question.strip()
    if not q:
        return JSONResponse(status_code=400, content={"error": "질문이 비어 있습니다."})

    t0 = time.time()
    try:
        with capture_path() as path:
            result = get_graph().invoke({"query": q})
    except Exception as e:  # 연결 거부(vLLM/ES 미기동) 등
        return JSONResponse(
            status_code=503,
            content={
                "error": "모델 서버 또는 검색엔진에 연결할 수 없습니다. "
                         "vLLM(8000)과 Elasticsearch(9200)가 떠 있는지 확인하세요.",
                "detail": str(e)[:300],
            },
        )

    docs = result.get("docs", [])
    stop = result.get("stop_reason", "n/a")
    return {
        "answer": result.get("answer", "(답변 없음)"),
        "stop_reason": stop,
        "stop_reason_kr": STOP_KR.get(stop, stop),
        "retry_count": result.get("retry_count", 0),
        "path": list(path),
        "sources": [
            {"index": i + 1, "preview": d[:120].replace("\n", " ")}
            for i, d in enumerate(docs[:5])
        ],
        "elapsed_sec": round(time.time() - t0, 2),
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

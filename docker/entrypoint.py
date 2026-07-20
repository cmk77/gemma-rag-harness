"""
docker/entrypoint.py — 데모 컨테이너 기동 시퀀스.

uvicorn을 바로 띄우지 않고, `docker compose up` 한 번으로 데모가 돌도록 준비를 마친다:
  1) Elasticsearch(ES_URL) 응답 대기
  2) 색인(ES_INDEX)이 없으면 corpus/ 자동 색인 — 첫 실행 시 bge-m3 다운로드(~2.3GB) 포함
  3) OLLAMA_URL이 설정돼 있으면 HEAVY_MODEL을 Ollama에 pull (이미 있으면 건너뜀)
  4) uvicorn web.api:app 으로 교체 실행(exec)

외부 vLLM 등 다른 OpenAI 호환 서버를 쓰는 경우 OLLAMA_URL을 비우면 3)을 건너뛴다.
각 단계는 실패해도 서버 기동은 막지 않는다(데모 우선) — 경고를 로그로 남긴다.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from http.client import HTTPResponse

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "gemma_rag")
CORPUS_DIR = os.getenv("CORPUS_DIR", "corpus")
OLLAMA_URL = os.getenv("OLLAMA_URL", "").rstrip("/")
MODEL = os.getenv("HEAVY_MODEL") or os.getenv("LLM_MODEL") or ""
PORT = os.getenv("PORT", "8080")
WAIT_SECS = int(os.getenv("STARTUP_WAIT_SECS", "180"))


def _log(msg: str) -> None:
    print(f"[entrypoint] {msg}", flush=True)


def _http(url: str, data: dict | None = None, timeout: float = 10.0) -> HTTPResponse:
    """의존성 없는 단순 GET/POST(JSON). 응답 객체를 반환한다."""
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def wait_for(name: str, url: str) -> bool:
    """서비스가 응답할 때까지 최대 WAIT_SECS초 대기."""
    deadline = time.time() + WAIT_SECS
    while time.time() < deadline:
        try:
            with _http(url, timeout=5):
                _log(f"{name} 준비됨: {url}")
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(3)
    _log(f"경고: {name}({url}) 응답 없음 — 계속 진행")
    return False


def ensure_index() -> None:
    """색인이 없으면 corpus/ 를 색인한다(이미 있으면 그대로 사용)."""
    try:
        with _http(f"{ES_URL}/{ES_INDEX}", timeout=5):
            _log(f"색인 '{ES_INDEX}' 존재 — 색인 건너뜀")
            return
    except urllib.error.HTTPError as e:
        if e.code != 404:
            _log(f"경고: 색인 확인 실패(HTTP {e.code}) — 색인 건너뜀")
            return
    except (urllib.error.URLError, OSError):
        _log("경고: ES 연결 불가 — 색인 건너뜀")
        return

    _log(f"색인 없음 → {CORPUS_DIR}/ 색인 시작 (첫 실행은 bge-m3 다운로드로 수 분 소요)")
    r = subprocess.run([sys.executable, "-m", "scripts.index_corpus", "--dir", CORPUS_DIR])
    if r.returncode != 0:
        _log("경고: 색인 실패 — 웹 서버는 뜨지만 검색 답변은 불가")


def ensure_ollama_model() -> None:
    """데모 LLM(Ollama) 모델을 준비한다. OLLAMA_URL이 없으면 아무것도 안 한다."""
    if not OLLAMA_URL or not MODEL:
        return
    if not wait_for("Ollama", f"{OLLAMA_URL}/api/version"):
        return
    try:
        with _http(f"{OLLAMA_URL}/api/tags") as res:
            names = {m.get("name", "") for m in json.load(res).get("models", [])}
        if MODEL in names:
            _log(f"Ollama 모델 '{MODEL}' 준비됨")
            return

        _log(f"Ollama 모델 '{MODEL}' 다운로드 시작 (수 GB — 수 분 걸릴 수 있음)...")
        last = ""
        with _http(f"{OLLAMA_URL}/api/pull", data={"name": MODEL}, timeout=3600) as res:
            for line in res:
                status = json.loads(line).get("status", "")
                if status and status != last:
                    _log(f"  {status}")
                    last = status
        _log(f"Ollama 모델 '{MODEL}' 준비 완료")
    except (urllib.error.URLError, OSError, ValueError) as e:
        _log(f"경고: Ollama 모델 준비 실패({e}) — 질의가 실패할 수 있음")


def main() -> None:
    wait_for("Elasticsearch", ES_URL)
    ensure_index()
    ensure_ollama_model()
    _log(f"웹 서버 기동: http://0.0.0.0:{PORT}")
    os.execvp("uvicorn", ["uvicorn", "web.api:app", "--host", "0.0.0.0", "--port", PORT])


if __name__ == "__main__":
    main()

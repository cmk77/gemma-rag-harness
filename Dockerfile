# ─────────────────────────────────────────────────────────────
# gemma-rag-harness 데모 이미지 — 하네스 + 웹 UI (CPU 전용)
#
# LLM 서빙은 이미지 밖에 둔다: 기본 데모는 docker-compose.yml의 Ollama,
# GPU 워크스테이션에선 HEAVY_URL만 vLLM(:8000/v1)으로 바꾸면 된다.
# 임베딩(bge-m3)은 첫 색인 때 HF에서 받는다(~2.3GB, hf-cache 볼륨에 저장).
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/cmk77/gemma-rag-harness" \
      org.opencontainers.image.description="자기검증형 RAG 에이전트 하네스 (LangGraph + Elasticsearch) 데모"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# torch는 CPU 휠로 먼저 고정 — 기본 인덱스의 CUDA 포함 휠(수 GB)을 피한다.
# sentence-transformers가 torch를 요구하지만 이미 충족돼 재설치되지 않는다.
RUN pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu

# 소스 복사(.dockerignore가 내부 문서·비밀을 차단) 후 editable 설치.
# editable이어야 prompts/·corpus/·web/static 등 저장소 상대 경로가 그대로 동작한다.
COPY . .
RUN pip install -e ".[web]"

EXPOSE 8080
ENTRYPOINT ["python", "docker/entrypoint.py"]

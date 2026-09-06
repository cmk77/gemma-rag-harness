# gemma-rag-harness

Gemma 4 E4B + MTP를 코어로 쓰는 **자기검증형 RAG 에이전트 하네스**.
LangGraph로 검색·생성·검증을 상태 기계로 묶고, 검증 실패 시 진단 기반 재검색을 돈다.

## 실행 ① — Docker (권장, GPU 불필요)

```bash
git clone https://github.com/cmk77/gemma-rag-harness.git
cd gemma-rag-harness
docker compose pull   # GHCR의 사전 빌드 이미지 다운로드 (생략하면 클론된 소스로 직접 빌드)
docker compose up -d
```

브라우저에서 **http://localhost:8080** 접속 → 우측 하단 챗 버튼.

- 사전 빌드 이미지: `ghcr.io/cmk77/gemma-rag-harness` — GitHub Actions가 main 푸시마다 발행.
  `pull`을 생략해도 compose가 같은 Dockerfile로 로컬 빌드하므로 동작은 동일하다(수 분 소요).
- 첫 실행은 다운로드로 시간이 걸린다(bge-m3 임베딩 ~2.3GB + Gemma 모델 ~3.3GB).
  진행 상황은 `docker compose logs -f app` 으로 확인.
- CPU 추론이라 답변에 수십 초 걸릴 수 있다. NVIDIA GPU가 있으면
  `docker-compose.yml`의 ollama GPU 주석을 해제하면 빨라진다.
- 더 가벼운 모델로 실행: `OLLAMA_MODEL=gemma3:1b docker compose up -d`
- 샘플 질문: `Elasticsearch 하이브리드 검색은 어떻게 융합하나요?`

## 실행 ② — 로컬 GPU 워크스테이션 (vLLM)

```bash
pip install -e ".[dev,web]" && pip install -e ".[serving]"   # serving(vLLM)은 GPU 머신 전용
bash serving/vllm_launch.sh        # 터미널1: Gemma 4 E4B+MTP 서빙 (:8000)
docker compose up -d es            # 터미널2: Elasticsearch + nori (:9200)
python -m scripts.index_corpus     # 코퍼스 색인 (첫 실행 시 bge-m3 다운로드)
python -m scripts.ask              # CLI 질의  (웹 UI: uvicorn web.api:app --port 8080)
```

단계별 설치 가이드: [SETUP.md](SETUP.md) (GPU 워크스테이션) · [SETUP_HOME.md](SETUP_HOME.md) (GPU 없는 노트북)

## 테스트 · 평가

```bash
ruff check . && pytest tests/                                    # 모델·ES 불필요
python -m eval.run_regression --goldenset eval/goldenset_sample.jsonl   # 골든셋 평가 (서빙+ES 필요)
```

---

개인 포트폴리오 프로젝트.

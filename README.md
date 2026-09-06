# gemma-rag-harness

Gemma 4 E4B + MTP를 코어로 쓰는 **자기검증형 RAG 에이전트 하네스**.
LangGraph로 검색·생성·검증을 상태 기계로 묶고, 검증 실패 시 진단 기반 재검색을 돈다.

## 개요

`harness/graph.py`가 정의하는 **6노드** 상태 기계다.

```
                    ┌──────────┐
                    │  ROUTER  │  질의 유형 판정
                    └────┬─────┘
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
      ┌────────┐    ┌────────┐    ┌────────┐
      │RETRIEVE│    │ DIRECT │    │ REJECT │
      └───┬────┘    └───┬────┘    └───┬────┘
          ▼             │             │
     ┌──────────┐       │             │
     │ GENERATE │       │             │
     └────┬─────┘       │             │
          ▼             │             │
     ┌──────────┐       │             │
     │  VERIFY  │       ▼             ▼
     └────┬─────┘      END           END
          │
   ok ────┴──── insufficient → 진단 기반 재검색 (RETRIEVE 재진입)
   ▼
  END
```

- **ROUTER** — 검색이 필요한 질의(RETRIEVE), 모델 지식만으로 답할 질의(DIRECT),
  코퍼스 범위 밖 질의(REJECT)로 분기한다.
- **RETRIEVE** — Elasticsearch 하이브리드 검색(BM25 + bge-m3 kNN, RRF 융합).
  재진입 시 `harness/adaptive_retrieve.py`가 k를 넓히고 이전 문서 풀에 누적한다.
- **GENERATE** — 검색된 근거만으로 답변을 생성한다.
- **VERIFY** — 근거 충실성(faithfulness)과 **질문-답변 주제 정합성**을 함께 판정한다.
  `insufficient`면 부족한 정보를 지목해 RETRIEVE로 되돌린다(상한까지).

제어 흐름(분기·순환·상한)은 그래프 정의 한곳에 있고, 노드 함수 안에 재시도 루프를
숨기지 않는다 — [ADR-0001](docs/adr/0001-langgraph-vs-plain-chain.md)의 결정이다.

## Quick Start (5분)

Docker만 있으면 된다. GPU 불필요.

```bash
git clone https://github.com/cmk77/gemma-rag-harness.git
cd gemma-rag-harness

# 1) 스택 기동 (Elasticsearch+nori · Ollama · 앱)
docker compose up -d

# 2) 샘플 코퍼스 색인 — 앱 컨테이너가 기동 시 ES를 기다렸다가
#    색인(ES_INDEX)이 없으면 자동으로 수행한다. 강제로 다시 돌리려면:
docker compose exec app python -m scripts.index_corpus

# 3) 질의 1건
curl -s localhost:8080/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"Elasticsearch 하이브리드 검색은 어떻게 융합하나요?"}'

# 4) 평가 배선 점검 (모델·ES 없이도 도는 dry-run)
docker compose exec app python -m eval.run_regression \
  --goldenset eval/goldenset_sample.jsonl --dry-run
```

브라우저로 보려면 **http://localhost:8080** → 우측 하단 챗 버튼.

> 첫 실행은 다운로드 때문에 5분보다 오래 걸린다(bge-m3 ~2.3GB + Gemma ~3.3GB).
> 진행 상황은 `docker compose logs -f app`.

## 실행 ① — Docker (권장, GPU 불필요)

```bash
docker compose pull   # GHCR의 사전 빌드 이미지 다운로드 (생략하면 클론된 소스로 직접 빌드)
docker compose up -d
```

- 사전 빌드 이미지: `ghcr.io/cmk77/gemma-rag-harness` — GitHub Actions가 main 푸시마다 발행.
  `pull`을 생략해도 compose가 같은 Dockerfile로 로컬 빌드하므로 동작은 동일하다(수 분 소요).
- CPU 추론이라 답변에 수십 초 걸릴 수 있다. NVIDIA GPU가 있으면
  `docker-compose.yml`의 ollama GPU 주석을 해제하면 빨라진다.
- 더 가벼운 모델로 실행: `OLLAMA_MODEL=gemma3:1b docker compose up -d`

## 실행 ② — 로컬 GPU 워크스테이션 (vLLM)

```bash
pip install -e ".[dev,web]" && pip install -e ".[serving]"   # serving(vLLM)은 GPU 머신 전용
bash serving/vllm_launch.sh        # 터미널1: Gemma 4 E4B+MTP 서빙 (:8000)
docker compose up -d es            # 터미널2: Elasticsearch + nori (:9200)
python -m scripts.index_corpus     # 코퍼스 색인 (첫 실행 시 bge-m3 다운로드)
python -m scripts.ask              # CLI 질의  (웹 UI: uvicorn web.api:app --port 8080)
```

단계별 설치 가이드: [SETUP.md](SETUP.md) (GPU 워크스테이션) · [SETUP_HOME.md](SETUP_HOME.md) (GPU 없는 노트북)

---

## 준비 상세

### 1) Elasticsearch + nori

nori(한국어 형태소 분석기)는 `Dockerfile.es`에 구워져 있다. compose를 쓰면 자동이다.

```bash
docker compose up -d es          # ghcr.io 이미지 또는 Dockerfile.es 로컬 빌드
```

nori가 실제로 붙었는지 확인(형태소로 쪼개져 나와야 정상):

```bash
curl -s localhost:9200 | grep cluster_name
curl -s -XPOST 'localhost:9200/_analyze' -H 'Content-Type: application/json' \
  -d '{"analyzer":"nori","text":"검색 엔진 하이브리드 색인"}'
```

compose 없이 직접 띄우려면:

```bash
docker build -f Dockerfile.es -t es-nori:latest .
docker run -d --name es -p 9200:9200 \
  -e discovery.type=single-node -e xpack.security.enabled=false \
  -e ES_JAVA_OPTS="-Xms1g -Xmx1g" es-nori:latest
```

### 2) bge-m3 임베딩 준비

별도 다운로드 단계는 없다. 첫 색인 때 `sentence-transformers`가 자동으로 받는다(약 2.3GB,
`~/.cache/huggingface`). 미리 받아두려면:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
```

> **버전 제약**: `sentence-transformers`는 반드시 **4.0 미만**이어야 한다. 5.x는 bge-m3 로딩 시
> `Pooling.__init__() missing 'embedding_dimension'` 로 실패한다. `pyproject.toml`이 `>=3.0,<4.0`으로
> 고정해 두었으므로 수동으로 올리지 말 것. vLLM이 transformers 5.x를 요구해 pip이 충돌 경고를 내지만,
> `sentence-transformers 3.4.x + transformers 5.12.x` 조합에서 bge-m3는 정상 동작한다.

Docker 데모는 `hf-cache` 볼륨에 캐시하므로 재기동 시 다시 받지 않는다.

### 3) 문서 ingest (원본 → corpus → ES)

색인 대상은 `corpus/*.md`, `corpus/*.txt`다. 원본 PDF/HTML이 있다면 먼저 마크다운으로 추출한다.

```bash
mkdir -p raw/pdf raw/html          # 원본을 여기에 둔다
python -m scripts.extract_docs                 # raw/ 전체 → corpus/*.md
python -m scripts.extract_docs --pdf-only      # PDF만 (pymupdf4llm, 표·레이아웃 보존)
python -m scripts.extract_docs --html-only     # HTML만 (trafilatura, 본문만 정제)
python -m scripts.extract_docs --preview raw/pdf/foo.pdf   # 한 파일 미리보기
```

> `corpus/`는 `.gitignore` 대상이다(`corpus/es_rag_guide.md` 샘플 1개만 예외).
> 어떤 문서를 넣더라도 커밋되지 않는다.

이어서 색인:

```bash
python -m scripts.index_corpus                 # corpus/ 전체 → ES
python -m scripts.index_corpus --dir mydocs    # 다른 디렉토리
```

토큰 단위 슬라이딩 윈도우로 청킹하며(`CHUNK_TOKENS=400`, `CHUNK_OVERLAP=80`),
청크 ID는 `sha1("{파일명}:{시작토큰}:{청크본문}")[:16]` 로 안정적으로 생성된다
(`retrieval/indexer.py`). 골든셋의 `gold_doc_ids`가 이 ID를 가리킨다.

### 4) 위키 코퍼스 파이프라인

기계적 청킹은 서로 다른 주제(예: '취득한 인증' vs '지원하는 인증 방식')를 한 청크에 섞어
검색 중의성을 만든다. 엔티티당 한 페이지 + 고정 섹션 스키마로 재구성하면 섹션이 곧 청크 경계가 된다.

```
corpus/*.md ──[A 추출]──▶ state/wiki/facts.jsonl ──[B 통합]──▶ ──[C 합성]──▶ corpus_wiki/*.md
```

vLLM 서빙이 떠 있어야 한다.

```bash
python -m scripts.build_wiki                   # corpus/ → corpus_wiki/
python -m scripts.build_wiki --limit 100       # 앞 100개만 (파일럿)
python -m scripts.build_wiki --workers 8       # 병렬 LLM 호출
python -m scripts.build_wiki --aliases aliases_example.json   # 표기 흔들림 정규화
python -m scripts.build_wiki --rebuild         # 체크포인트 무시 전체 재빌드
```

문서 sha1 기반 증분 처리 + JSONL append 체크포인트라 중단해도 이어서 실행된다.

A/B로 효과 검증:

```bash
ES_INDEX=gemma_rag_wiki python -m scripts.index_corpus --dir corpus_wiki
ES_INDEX=gemma_rag_wiki python -m eval.run_regression --goldenset eval/goldenset_sample.jsonl
# 기존 인덱스(gemma_rag) 결과와 비교
```

개인 지식 볼트용 CLI도 같은 엔진을 쓴다:

```bash
python -m wiki_vault.cli onboard --name "사용자명" --interests "RAG, LangGraph"
python -m wiki_vault.cli ingest        # 인박스 → 카드
python -m wiki_vault.cli query "질문"
python -m wiki_vault.cli audit         # 볼트 전수 감사
```

### 5) Judge 평가

`.env`에 백엔드를 먼저 설정한다(`cp .env.example .env`). Judge는 서빙 중인 모델을 그대로 쓴다.

```bash
# 배선만 점검 — 모델·ES 불필요
python -m eval.run_regression --goldenset eval/goldenset_sample.jsonl --dry-run

# 실제 평가 (vLLM + ES 필요)
python -m eval.run_regression --goldenset eval/goldenset_sample.jsonl \
  --out eval/reports/latest.json

# 회귀 게이트 — 핵심 지표가 베이스라인 대비 하락하면 exit 1
python -m eval.run_regression --goldenset eval/goldenset_sample.jsonl \
  --baseline eval/reports/baseline.json
```

회귀 임계치: `Recall@5` −0.05, `MRR` −0.05, `faithfulness` −0.3, `correctness` −0.3.

일관성(중의성 구분 · 표현 불변) 검증:

```bash
python -m eval.run_consistency --repeat 3
python -m eval.run_consistency --pairs eval/consistency_pairs_sample.jsonl --repeat 5
```

- invariant 페어: 표현이 달라도 같은 정답으로 수렴하면 PASS
- discriminative 페어: 중의적인 두 질문이 서로 다른 올바른 답을 내면 PASS

> 골든셋을 새로 만들 때는 `gold_doc_ids`를 반드시 채운다. 비어 있으면
> `retrieval_eval.load_goldenset`이 해당 문항을 검색 평가에서 제외해
> Recall@5·MRR·nDCG가 항상 0으로 집계된다.

---

## 테스트 · 평가

```bash
ruff check . && pytest tests/                                          # 모델·ES 불필요
python -m eval.run_regression --goldenset eval/goldenset_sample.jsonl  # 골든셋 평가 (서빙+ES 필요)
```

현재 main 기준 **`ruff check .` 통과 · `pytest tests/` 42 passed**.
테스트는 `tests/conftest.py`가 `langchain_openai`·`langgraph`·ES를 스텁으로 주입해
모델과 Elasticsearch 없이 돈다. CI(`.github/workflows/ci.yml`)도 같은 두 명령에
`run_regression --dry-run`을 더해 돌린다.

## 프로젝트 구조

```
harness/      그래프 정의(graph.py) · 적응형 재검색 · 쿼리 확장 · 프롬프트 로더
retrieval/    VectorStore 인터페이스 · ES 구현 · 색인/청킹 · RRF 융합
models/       LLM 백엔드 추상화 (backends.py — vLLM/Ollama 무변경 전환)
prompts/      시스템 프롬프트 (router·generate·verify·hyde·rewrite·decompose 등 .md)
eval/         골든셋 · 검색 지표 · LLM-as-Judge · 회귀/일관성 러너
scripts/      ask · index_corpus · extract_docs · build_wiki
wiki/         엔티티 위키 스키마 · 사실 추출 · 페이지 합성
wiki_vault/   개인 지식 볼트 CLI (같은 검색·검증 엔진 재사용)
web/          FastAPI 플로팅 위젯(api.py + static/) · Streamlit 챗(streamlit_app.py)
serving/      vllm_launch.sh (Gemma 4 E4B + MTP)
docker/       컨테이너 entrypoint
tests/        단위 테스트 (모델·ES 없이 도는 것만)
docs/adr/     아키텍처 결정 기록
```

### ADR

| # | 결정 |
|---|---|
| [0001](docs/adr/0001-langgraph-vs-plain-chain.md) | 하네스 오케스트레이션 — LangGraph vs plain LangChain Chain |
| [0002](docs/adr/0002-rrf-hybrid-search.md) | 하이브리드 검색 융합 — RRF vs 가중합(Weighted Score) |
| [0003](docs/adr/0003-single-model-multi-role.md) | 노드별 모델 전략 — 단일 모델 다역할 vs 역할별 전용 모델 |
| [0004](docs/adr/0004-retrieval-k-and-verify-hardening.md) | 검색 폭(BASE_K) 상향 및 VERIFY 프롬프트 강화 |

## 데이터 정책

저장소에는 **공개 샘플 문서 1건(`corpus/es_rag_guide.md`)과 샘플 골든셋만** 포함한다.
`corpus/`, `corpus_wiki/`, `state/`, `raw/` 는 `.gitignore` 대상이며 사용자 문서는 커밋되지 않는다.

| 경로 | 상태 |
|---|---|
| `corpus/` | 무시됨 — `corpus/es_rag_guide.md` 샘플 1건만 예외 허용 |
| `raw/` | 무시됨 — 원본 PDF/HTML 보관 위치 |
| `corpus_wiki/` | 무시됨 — 위키 파이프라인 산출물 |
| `state/` | 무시됨 — 증분·체크포인트 JSONL |
| `eval/goldenset_sample.jsonl` | 커밋됨 — 샘플 코퍼스만으로 답할 수 있는 5문항 |
| `eval/consistency_pairs_sample.jsonl` | 커밋됨 — 일관성 페어 3쌍 |
| `eval/reports/latest.json` | 무시됨 — 실행 산출물 |

실제 문서 코퍼스로 평가하려면 `--goldenset`·`--baseline`에 로컬 경로를 직접 지정한다.
저장소에 커밋하지 않는 것을 전제로 설계돼 있다.

---

개인 포트폴리오 프로젝트.

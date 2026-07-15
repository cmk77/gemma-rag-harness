# gemma-rag-harness

> Gemma 4 E4B + MTP(추측 디코딩)를 코어 오케스트레이터로 쓰는 **자기검증형 RAG 에이전트 하네스**.
> LangGraph로 검색·생성·검증을 상태 기계로 묶고, 검증 실패 시 진단을 근거로
> 재검색하는 루프를 갖췄다. 검색·생성 품질을 골든셋으로 측정하고 CI에서 회귀를 막는다.

**스택**: Gemma 4 E4B + MTP · LangGraph/LangChain · Elasticsearch(하이브리드 검색) · vLLM(서빙) · LLM-as-Judge(평가)

> **처음 실행하나요?** WSL 환경 단계별 설치·실행 가이드는 [SETUP.md](SETUP.md)를 보세요.
> 집 노트북 등 **Anaconda 기반 세팅**은 [SETUP_HOME.md](SETUP_HOME.md)를 보세요.

---

## 왜 이 프로젝트인가 — 채용 요건 매핑

이 저장소는 ㈜인포유앤컴퍼니 AI 엔지니어 공고의 자격요건을 **동작하는 코드로 증빙**하기 위해 만들었다.

| 공고 자격요건 | 구현 위치 | 비고 |
|---|---|---|
| Elasticsearch 등 벡터 DB 기반 RAG 구현 | `retrieval/es_store.py` | BM25 + dense kNN, RRF 융합 |
| LangChain/LlamaIndex 기반 AI 에이전트 개발 | `harness/graph.py` | LangGraph 상태 기계 |
| LLM 성능 최적화(토큰/속도/정확도) + 운영 | `serving/` + `eval/` | vLLM, 프롬프트 캐싱, 레이턴시 계측 |
| 프롬프트 개선 + 생성형 AI 품질 검증 | `eval/judge.py`, `prompts/` | LLM-as-Judge 3축 채점, 프롬프트 버전관리 |
| Hugging Face/Ollama 오픈모델 활용 | `models/backends.py` | vLLM/Ollama/HF 백엔드 추상화 |
| Git 협업 + 코드 리뷰 | `.github/` | CI(lint·test·회귀), PR 템플릿 |
| AI Coding Assistant 실무 활용 | `CLAUDE.md` + 커밋 이력 | Claude Code 기반 개발 |
| 요구사항을 기술 문서로 구체화 | `docs/` | 기획서 + ADR 3건 |

설계 의사결정의 근거는 [`docs/adr/`](docs/adr/)에 ADR로 기록했다:
RRF 채택, LangGraph 채택, 단일 모델 다역할 — 각각 대안 비교와 trade-off 포함.

---

## 아키텍처

### 하네스 그래프 (LangGraph)

```
                    ┌─────────────┐
   user query ─────▶│   ROUTER    │  Gemma: 질의 유형 분류
                    └──────┬──────┘
              ┌────────────┼────────────┐
              ▼            ▼             ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ RETRIEVE │ │  DIRECT  │ │  REJECT  │
        │(하이브리드│ │(검색불요) │ │(범위 밖)  │
        │  검색)    │ └────┬─────┘ └──────────┘
        └────┬─────┘      │
             ▼            │
        ┌──────────┐      │
        │ GENERATE │◀─────┘  Gemma: 근거 기반 답변
        └────┬─────┘
             ▼
        ┌──────────┐
        │  VERIFY  │  Gemma: 근거 충실성 self-check + 진단
        └────┬─────┘
             │ insufficient → 진단 기반 재검색 (RETRIEVE 재진입)
             │   · retry별 전략 에스컬레이션(rewrite→HyDE→decompose)
             │   · 루프 가드: 새 문서 0개면 조기 종료
             │ ok → END
             ▼
        답변 + 출처 + stop_reason
```

### 검색 계층 (하이브리드 + RRF)

```
질의 ──┬─ BM25 (nori 형태소, 키워드)  ──┐
       └─ kNN  (bge-m3 임베딩, 의미)  ──┴─ RRF 융합 ─→ 상위 k 청크
```

### 평가 계층

```
골든셋 ──┬─ retrieval_eval : Recall@k / MRR / nDCG  (검색 단독)
         └─ judge          : faithfulness / correctness / relevance  (생성 품질)
                              ↓
                     run_regression → 리포트 + 베이스라인 대비 회귀 감지 → CI 게이트
```

---

## 빠른 시작

### 1. 설치

```bash
git clone <repo-url> && cd gemma-rag-harness
pip install -e .
```

### 2. Gemma 4 E4B + MTP 서빙 (vLLM)

```bash
# Gemma 4는 Apache-2.0 공개 모델 — 토큰 불필요
bash serving/vllm_launch.sh
# OpenAI 호환 엔드포인트가 http://localhost:8000/v1 에 뜬다
```

> 상세한 설치·실행(GPU 확인, 토큰 발급, 색인, 재시작 순서)은 [SETUP.md](SETUP.md) 참고.
> `models/backends.py`는 vLLM/Ollama/HF를 추상화하지만, 이 프로젝트의 표준 경로는 vLLM이다.

### 3. Elasticsearch 기동 + 색인

```bash
# ES 8.x가 http://localhost:9200 에 떠 있다고 가정
python -m retrieval.indexer    # 샘플 코퍼스 색인 (실제 코퍼스로 교체)
```

### 4. 질의 실행

```bash
python -m harness.graph
# 또는 코드에서:
#   from harness.graph import build_graph
#   app = build_graph()
#   result = app.invoke({"query": "Elasticsearch 하이브리드 검색은 어떻게 융합하나요?"})
#   print(result["answer"], result["stop_reason"])
```

### 5. 평가 + 회귀 체크

```bash
# 모델·ES 없이 배선만 점검
python -m eval.run_regression --goldenset eval/goldenset.jsonl --dry-run

# 실제 평가 (서빙 + ES 필요) — 베이스라인 대비 회귀 시 exit 1
python -m eval.run_regression \
  --goldenset eval/goldenset.jsonl \
  --baseline eval/reports/baseline.json
```

---

### 웹 챗봇 UI (선택)

CLI 대신 웹에서 쓰려면 두 가지 방식이 있다 (`pip install -e ".[web]"` 선행):

**① FastAPI — 플로팅 위젯** (페이지 우측 하단 챗 버튼 → 클릭 시 채팅창)
```bash
uvicorn web.api:app --host 0.0.0.0 --port 8080
# → http://localhost:8080  (vLLM·ES가 떠 있어야 답변 가능)
```

**② Streamlit — 전체 페이지 챗** (빠른 데모·내부 테스트용)
```bash
streamlit run web/streamlit_app.py
# → http://localhost:8501
```

두 UI 모두 답변과 함께 **근거 문서**·**실행 경로(ROUTER→RETRIEVE→GENERATE→VERIFY)**를
펼쳐 보여준다. 포트: vLLM=8000, FastAPI=8080, Streamlit=8501.


### 위키 코퍼스 파이프라인 (선택 — 검색 정확도 개선)

기계적 청킹 대신 **엔티티 중심 위키 페이지**로 코퍼스를 재구성한다.
'취득 인증·수상'과 '지원 인증·규격'이 스키마 차원에서 분리돼 검색 중의성이 줄어든다.
수천~수백만 문서 전제로 설계: map/reduce 3단계, 문서 sha1 기반 증분, 체크포인트
재개, `--workers` 병렬, 파싱 실패 격리(errors.jsonl).

```bash
python -m scripts.build_wiki --workers 8          # corpus/ → corpus_wiki/
# A/B 검증: 별도 인덱스에 색인해 골든셋 점수 비교
ES_INDEX=gemma_rag_wiki python -m scripts.index_corpus --dir corpus_wiki
ES_INDEX=gemma_rag_wiki python -m eval.run_regression --goldenset eval/goldenset.jsonl
```


## 저장소 구조

```
gemma-rag-harness/
├── CLAUDE.md                  # Claude Code 작업 규약
├── README.md
├── pyproject.toml
├── docs/
│   ├── 기획서.md
│   └── adr/                   # 아키텍처 결정 기록 (대안 비교 + trade-off)
│       ├── 0001-langgraph-vs-plain-chain.md
│       ├── 0002-rrf-hybrid-search.md
│       └── 0003-single-model-multi-role.md
├── harness/
│   ├── graph.py               # LangGraph StateGraph (코어)
│   ├── adaptive_retrieve.py   # 진단 기반 재검색 + 루프 가드
│   └── query_expansion.py     # rewrite / decompose / HyDE
├── retrieval/
│   ├── vector_store.py        # VectorStore 인터페이스 (ES/Cosmos 추상화)
│   ├── es_store.py            # Elasticsearch 구현
│   ├── indexer.py             # 청킹 · 임베딩 · 색인
│   └── fusion.py              # RRF
├── models/
│   └── backends.py            # vLLM / Ollama / HF 추상화
├── serving/
│   ├── vllm_launch.sh
│   └── metrics.py             # 토큰 · 레이턴시 계측
├── eval/
│   ├── goldenset.jsonl
│   ├── retrieval_eval.py      # Recall@k / MRR / nDCG
│   ├── judge.py               # LLM-as-Judge
│   └── run_regression.py      # CI 회귀 게이트
├── prompts/                   # 버전관리되는 시스템 프롬프트
└── .github/
    ├── workflows/ci.yml
    └── pull_request_template.md
```

---

## 핵심 설계 특징

- **자기검증 + 진단 기반 재검색**: 단순 RAG가 아니라, 답변을 검증하고 부족하면
  *무엇이* 부족한지 진단해 그걸로 쿼리를 재작성한 뒤 재검색한다. 같은 쿼리로
  k만 늘리는 순진한 재시도와 다르다.
- **루프 가드**: 재검색이 새 문서를 못 가져오면 상한 도달 전에 조기 종료하고
  종료 사유(`stop_reason`)를 남긴다 — 운영 디버깅 용이.
- **측정 가능성**: 검색과 생성을 분리 평가. 측정 실패(Judge 파싱 깨짐)를 0점이
  아닌 `None`으로 격리해 회귀 판정 오염을 막는다.
- **교체 가능성**: 검색 백엔드(ES/Cosmos)와 모델 백엔드(vLLM/Ollama/HF)를
  인터페이스 뒤에 숨겨 환경에 따라 교체 가능.

---

## 라이선스

개인 포트폴리오 프로젝트.

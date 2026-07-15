# CLAUDE.md

이 파일은 Claude Code가 `gemma-rag-harness` 저장소에서 작업할 때 따르는 규약이다.
사람 기여자에게도 동일하게 적용되는 프로젝트 컨벤션이다.

## 프로젝트 한 줄 요약

Gemma 4 E4B + MTP(추측 디코딩)를 코어로 쓰는 자기검증형 RAG 에이전트 하네스.
LangGraph로 검색·생성·검증을 상태 기계로 묶고, 검증 실패 시 진단 기반 재검색을 돈다.

## 현재 상태 스냅숏 (2026-07-06 기준 — claude.ai 채널에서 확정된 사항)

이 섹션은 웹 채널("Gemma 4 기반 LangGraph 하네스" 대화)에서 결정·검증된 내용의 요약이다.
Claude Code로 작업을 이어받을 때 이 스냅숏을 사실로 전제하라.

### 확정된 구성
- **모델**: Gemma 4 **E4B + MTP**(추측 디코딩) 단일 모델, vLLM 0.24, 포트 8000.
  `serving/vllm_launch.sh`가 `--speculative-config method=mtp`로 초안 모델
  (`google/gemma-4-E4B-it-assistant`, 0.15GB, KV 공유)을 붙인다. VRAM ~22GB/48GB.
  (이력: Gemma 3 27B 단일 → 27B+4B 2티어는 VRAM 배분 실패로 폐기 → E4B+MTP 확정)
- **버전 시소 해소(실측)**: sentence-transformers 3.4.x + transformers 5.12.x 조합에서
  bge-m3·vLLM 모두 정상. pip 충돌 경고는 무해. pyproject에 `sentence-transformers>=3.0,<4.0` 핀.
- **환경**: 워크스테이션 = WSL2 + venv + RTX 6000 Ada 48GB. 집 노트북 = WSL2 + conda,
  NVIDIA 없음 → Ollama(CPU) 경로 B (SETUP_HOME.md). `.env` 두 줄로 서빙 전환 가능.

### 해결된 핵심 이슈 — GATEWAY-A "인증" 중의성 (오독 금지)
- "받은/취득한 인증"(certification → **GS인증 1등급**)과 "지원하는 인증"
  (authentication → Basic·API Key·JWT·OAuth2)은 **다른 질문, 다른 정답**이다.
- 과거 오답 원인: 벡터 성분이 "인증" 중의성으로 authentication 문서를 top-5에
  올림 + VERIFY가 질문을 안 봐서 표현에 따라 판정이 갈림.
- **적용된 수정**: `BASE_K 5→10`, `MAX_CONTEXT_DOCS 12→15`(adaptive_retrieve.py),
  verify 노드가 [질문]을 프롬프트에 전달(graph.py), verify.md에 질문-답변
  주제 정합성 검사 명시. 두 질문 유형 모두 재검색 0회로 각각 정답 → **개선 성공 상태**.
- ES 검색 필드는 `content`가 아니라 **`text`**다 (진단 시 헷갈리지 말 것).

### 평가 체계
- ExampleCorp 골든셋 15문항(eval/goldenset.jsonl). 베이스라인: faithfulness 4.733 /
  correctness 4.533 / relevance 4.867. 개선 후 재평가로 correctness 상승 확인이 남은 일.
- 확장 산출물(50문항 golden_50.jsonl, 일관성 페어, ADR-0001)이 별도 생성돼
  있으며 eval/·docs/adr/ 반영 여부는 작업 전 디렉토리를 확인하라.

### 웹 프론트엔드 (web/)
- FastAPI(`uvicorn web.api:app --port 8080`) — 우측 하단 플로팅 챗 위젯 + 근거·실행경로 표시.
- Streamlit(`streamlit run web/streamlit_app.py`, 8501) — 전체 페이지 챗.
- 실행경로는 harness 로거 캡처(web/capture.py) — 락으로 질의 직렬화(데모 기준).

### 진행 중 로드맵
1. **위키 코퍼스 ①**: `python -m scripts.build_wiki --workers 4` →
   `ES_INDEX=gemma_rag_wiki python -m scripts.index_corpus --dir corpus_wiki` →
   같은 골든셋으로 A/B (기존 correctness 4.533 대비). ← **다음 실행 대기**
2. ① 효과 확인 후 **② 소규모 지식그래프**(트리플 수십 개, build_wiki 골격 재사용).
3. 수천~수백만 문서 전제 유지: 증분(sha1)·체크포인트·병렬(--workers)·오류 격리 설계를 깨지 마라.

### 실행 표준 (매번)
```bash
# 터미널1: bash serving/vllm_launch.sh     # E4B+MTP, 8000
# 터미널2: docker start es && python -m scripts.ask   # 또는 uvicorn web.api:app --port 8080
# 설정 확인: python -m models.backends  → gemma-4-E4B-it @ localhost:8000/v1
# (8002가 나오면 셸에 옛 환경변수 잔재: unset HEAVY_URL HEAVY_MODEL LIGHT_URL LIGHT_MODEL)
```

## 아키텍처 불변식 (깨지 말 것)

이 규칙들은 ADR로 결정된 것이다. 바꾸려면 먼저 해당 ADR을 갱신하라.

1. **단일 모델 다역할** (ADR-0003). ROUTER/GENERATE/VERIFY/쿼리확장은 같은 모델을
   *다른 시스템 프롬프트로* 호출한다. 노드마다 다른 모델을 박지 마라. 모델 접근은
   반드시 `models/backends.py`의 `get_llm()` / `ask()`를 거친다 — `ChatOpenAI`를
   노드에서 직접 생성하지 마라.
2. **검색 백엔드는 인터페이스 뒤에** (ADR-0002 관련). 하네스는 `retrieval/vector_store.py`의
   `VectorStore` 타입에만 의존한다. ES 전용 코드를 `harness/`에 새지 마라.
3. **오케스트레이션은 LangGraph** (ADR-0001). 제어 흐름(분기·순환·상한)은
   `harness/graph.py`의 그래프 정의 한곳에 둔다. 노드 함수 안에 while 루프로
   재시도를 숨기지 마라.
4. **프롬프트는 `prompts/*.md`에 외부화**. 시스템 프롬프트를 코드에 하드코딩하지 마라.
   `harness/prompts.py`의 로더로 읽는다.
5. **측정 실패 ≠ 0점**. 평가에서 파싱 실패는 `None`으로 격리하고 집계에서 제외한다.
   0점으로 처리하면 회귀 판정이 오염된다.

## 코드 스타일

- Python 3.11+, 타입힌트 필수. `from __future__ import annotations`를 파일 상단에.
- 린트는 ruff (`ruff check .`). 설정은 `pyproject.toml`.
- 한국어 주석/독스트링 유지(이 프로젝트의 관례). 변수·함수명은 영문.
- 함수는 가능한 순수하게 — 특히 그래프 노드는 `state in → dict out` 형태를 지킨다
  (테스트 용이성). 부수효과(I/O, 전역 상태)는 최소화.
- 새 의존성 추가는 신중히. 추가 시 `pyproject.toml`과 이 파일에 이유를 남긴다.

## 작업 방식

- **변경 전 관련 ADR을 읽어라.** 아키텍처에 영향 주는 변경은 ADR 갱신/추가가 선행한다.
- **테스트 먼저 돌려라.** `pytest tests/`는 모델·ES 없이 통과해야 한다(스텁 사용).
  모델이 필요한 검증은 `--dry-run` 또는 self-hosted 러너로 분리.
- **작은 PR.** 한 PR은 한 가지 관심사. 스프린트 경계(S1~S4)를 따른다.
- **커밋 메시지는 Conventional Commits.** `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`.
  예: `feat(harness): add diagnosis-guided requery loop`
- **회귀를 숫자로 증명.** 검색/생성 품질에 영향 주는 변경은 `eval/run_regression.py`
  결과를 PR 본문에 첨부한다(베이스라인 대비 Δ).

## 자주 쓰는 명령

```bash
# 린트 + 테스트 (모델 불필요)
ruff check . && pytest tests/

# 평가 배선 점검 (모델 불필요)
python -m eval.run_regression --goldenset eval/goldenset.jsonl --dry-run

# 서빙 기동 (GPU 필요)
bash serving/vllm_launch.sh

# 질의 실행 (서빙 + ES 필요)
python -m scripts.ask
```

## 디렉토리 지도

- `harness/` — 그래프(graph.py), 재검색(adaptive_retrieve.py), 쿼리확장, 프롬프트 로더
- `retrieval/` — VectorStore 인터페이스 + ES 구현 + 색인 + RRF
- `models/` — 백엔드 추상화 (vllm/ollama/hf)
- `eval/` — 골든셋, 검색평가, Judge, 회귀러너
- `prompts/` — 시스템 프롬프트 (md)
- `docs/adr/` — 아키텍처 결정 기록
- `tests/` — 단위 테스트 (모델 없이 도는 것만)

## 하지 말 것 (요약)

- 노드에서 `ChatOpenAI` 직접 생성 → `get_llm()` 사용
- `harness/`에 ES 전용 코드 → `retrieval/` 뒤로
- 프롬프트 하드코딩 → `prompts/*.md`
- 노드 함수에 재시도 루프 숨기기 → 그래프 엣지로
- 평가 파싱 실패를 0점 처리 → `None` 격리
- 모델 없이는 못 도는 테스트를 `tests/`에 추가 → 스텁화하거나 분리

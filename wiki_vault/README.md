# wiki_vault — LLM Wiki 지식 볼트 모듈

강연 "AI 에이전트를 위한 지식 베이스 구축(옵시디언 LLM Wiki)"의 핵심 개념을
**gemma-rag-harness에 그대로 마운트 가능한 독립 패키지**로 코드화한 모듈.
마크다운 볼트가 소스오브트루스, Elasticsearch는 색인 사본이며,
옵시디언으로 볼트 폴더를 열면 그래프 뷰/링크가 그대로 동작한다.

## 1. 개념 → 모듈 매핑

| 강연 개념 | 구현 | 파일 |
|---|---|---|
| 듀얼 볼트 (모선=메인 / 위성=위키) | `VaultManager` — 위키 볼트만 쓰기, 메인 볼트는 읽기 전용 폴백 스캔 | `vault.py` |
| YAML 메타데이터 | `Frontmatter` (title/description 영어, tags, status, model 출처, source) | `models.py` |
| Progressive Disclosure | S1 메타 전용 검색 → LLM이 메타만 보고 열어볼 카드 선별 → S2 상위 k만 본문 로드(문자 예산) | `indexer.py` + `query.py` |
| Ingest (인박스→카드) | `0_inbox` 버퍼 → LLM 그룹핑(`/inbox`) → Concept/Entity 카드 → 로우 소스 `90_raw` 이동 | `ingest.py` |
| #목적 해시태그 | 인박스 원문에서 추출해 카드 tags로 강제 승계 | `ingest.py` |
| 논문 초록 모드 | `--abstract-only` (전문 인제스트 토큰 낭비 방지) | `ingest.py` |
| Query (BM25/벡터/가설검증) | 메타 BM25 → 본문 BM25+kNN 수동 RRF → 초안 → VERIFY → 실패 시 wide_k 재검색 1회 | `query.py` |
| 모선 폴백 | 위키 근거 부족 시 메인 볼트 키워드 스캔 | `vault.py` + `query.py` |
| Lint / Audit / Verify | 신규 카드 유입 훅(구지식 update/outdated/교차링크), 전수 감사, 개별 사실성 재점검 | `lint.py` |
| corecontext / rules | `_system/corecontext.md`, `_system/rules/*.md` — 모든 LLM 호출 시스템 프롬프트 선두에 주입 | `vault.py` + `cli.py onboard` |
| 작성 모델 명시 | 카드 `model:` 필드에 생성 모델 자동 기록 | `ingest.py` |
| 쿼리 히스토리 | `query --save` → `30_notes`에 산출물 카드 저장 | `query.py` |

## 2. 하네스 재사용 지점

- LLM: OpenAI 호환 (`VLLM_URL`/`LLM_MODEL` 두 변수 — vLLM E4B+MTP ↔ 집 노트북 Ollama 무변경 전환)
- ES: nori 분석기, 본문 필드명 **`text`** 고정(과거 content 오진단 교훈), BM25+kNN 수동 RRF
- 임베딩: bge-m3(1024d). sentence-transformers 미설치 시 BM25-only 자동 강등
- VERIFY: ADR-0001 교훈 이식 — [질문]을 검증 프롬프트에 명시 전달 + 주제 정합성 검사('받은 인증' vs '지원 인증'류 중의성 차단), 실패 시 k 상향 재검색

## 3. 설치

```bash
# gemma-rag-harness 루트에 wiki_vault/ 폴더 복사 후
pip install pyyaml httpx elasticsearch      # 대부분 이미 설치됨
# 선택: sentence-transformers (bge-m3 kNN)  — 하네스 핀 유지: >=3.0,<4.0
```

`.env` 추가분 (기존 변수는 그대로 재사용):

```
WIKI_VAULT_ROOT=./vault_wiki
MAIN_VAULT_ROOT=/mnt/c/Users/mozi/ObsidianMain   # 선택(모선 폴백)
WIKI_ES_INDEX=gemma_rag_wiki_vault
ES_URL=http://localhost:9200
# VLLM_URL / LLM_MODEL / EMBED_MODEL 은 기존 값 공유
```

## 4. 퀵스타트 (강연 실습 흐름)

```bash
python -m wiki_vault.cli onboard --name "Mozi" --org "ExampleCorp" \
    --interests "RAG, LangGraph, Elasticsearch"
# → vault_wiki/ 에 0_inbox / 10_concepts / 20_entities / 30_notes /
#   90_raw / _system(corecontext.md, rules/) 생성. 옵시디언으로 열기 가능.

# 웹클리퍼/스크랩을 0_inbox에 넣고 (#목적 태그 함께 기입 권장)
python -m wiki_vault.cli inbox                    # 그룹핑 제안 미리보기
python -m wiki_vault.cli ingest --commentary "E4B 하네스 검색 개선 참고용"
python -m wiki_vault.cli ingest --abstract-only   # 논문 배치일 때

python -m wiki_vault.cli query "RRF 융합이 중의성 질의에 미치는 영향 정리" --save
python -m wiki_vault.cli audit
python -m wiki_vault.cli reindex                  # 마이그레이션/복구
```

## 5. 하네스 LangGraph 통합

방법 A — ROUTER 분기 함수 (최소 변경):

```python
# harness/graph.py
from wiki_vault.bootstrap import build_default_engine
from wiki_vault.graph import wiki_query_tool

_wiki_q = wiki_query_tool(build_default_engine())

def router(state):
    ...
    if route == "wiki":          # 개인 지식/스크랩 계열 질의
        return _wiki_q(state["question"])
```

방법 B — 서브그래프 마운트:

```python
from wiki_vault.bootstrap import build_app
from wiki_vault.graph import build_wiki_graph

wiki_graph = build_wiki_graph(build_app().query)
graph.add_node("WIKI", wiki_graph)   # ROUTER 조건부 엣지에 "WIKI" 추가
```

웹UI(web/api.py)에서도 `build_default_engine().run(q)` 한 줄로 호출 가능
(trace가 기존 '실행경로' 표시 포맷과 동일한 list[str]).

## 6. 카드 포맷 예시

```markdown
---
title: RRF Fusion
description: Reciprocal Rank Fusion for hybrid BM25+kNN retrieval.
type: concept
tags: ["#purpose/harness-retrieval", "search"]
status: active
created: 2026-07-05
updated: 2026-07-05
model: gemma-4-E4B-it
source: article-rrf.md
links: ["10_concepts/BM25.md"]
vault: wiki
---

## 개요
...

## 핵심 사실
...

## 맥락/코멘터리
(사용자 코멘터리 — 인제스트 시 --commentary로 주입)
```

## 7. 기존 scripts/build_wiki.py 와의 관계

- `build_wiki.py`(①로드맵) = **검색 코퍼스 재구성** 파이프라인(엔티티 페이지 합성, ES 색인 전용)
- `wiki_vault`(본 모듈) = **개인 지식 볼트 계층**(마크다운 소스오브트루스 + 옵시디언 호환 + Ingest/Query/Lint 라이프사이클)
- 상호보완: wiki_vault 카드가 쌓이면 `reindex` 인덱스를 A/B용 코퍼스
  (`ES_INDEX=gemma_rag_wiki_vault`)로 그대로 회귀 평가에 투입 가능.
  ②지식그래프 착수 시 카드 `links:`가 트리플 후보 소스가 된다.

## 8. 파일 구성

```
wiki_vault/
├── __init__.py      # 공개 API
├── config.py        # env 설정 (하네스 컨벤션 공유)
├── models.py        # Frontmatter / WikiCard / 점진적 노출 뷰
├── vault.py         # 듀얼 볼트, 인박스, 링크 변환, 모선 스캔
├── llm.py           # OpenAI 호환 클라이언트 (+JSON 강제)
├── embeddings.py    # bge-m3 (미설치 시 BM25-only 강등)
├── indexer.py       # ES 메타/본문 색인, BM25·kNN·RRF
├── ingest.py        # Ingest 파이프라인 (그룹핑/초록모드/태그승계)
├── query.py         # Query 엔진 (3단계 + VERIFY + 모선 폴백)
├── lint.py          # Lint/Audit/Verify
├── graph.py         # LangGraph 서브그래프 + ROUTER 도구
├── bootstrap.py     # 조립 팩토리 / 전수 재색인
└── cli.py           # onboard·inbox·ingest·query·lint·audit·verify·reindex
```

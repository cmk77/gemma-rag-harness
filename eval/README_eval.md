# 평가 산출물 사용 안내

`eval/`의 평가 자산과 실행법을 정리한다. 모든 명령은 **프로젝트 루트**에서 실행한다.

## 파일 목록

| 파일 | 용도 |
|---|---|
| `goldenset_sample.jsonl` | 공개 샘플 골든셋 5문항. `corpus/es_rag_guide.md` 만으로 답할 수 있고 `gold_doc_ids`에 실제 청크 ID가 채워져 있다 |
| `consistency_pairs_sample.jsonl` | 일관성 검증 페어 3쌍(invariant 2 + discriminative 1) |
| `run_regression.py` | 골든셋 평가 + 회귀 게이트 러너 |
| `run_consistency.py` | 일관성 검증 러너 |
| `retrieval_eval.py` | 검색 단독 지표(Recall@k · MRR · nDCG@k · Hit@k) |
| `judge.py` | LLM-as-Judge 채점 |
| `../docs/adr/0004-retrieval-k-and-verify-hardening.md` | BASE_K↑ + VERIFY 강화 결정 기록 |

> 실 코퍼스 골든셋과 베이스라인 리포트는 공개 저장소에 두지 않는다.
> 사내 평가는 `--goldenset`·`--baseline`에 로컬 경로를 직접 지정해 돌린다.

## 1. 골든셋 평가

```bash
# 서빙·ES 기동 (표준 순서)
bash serving/vllm_launch.sh            # 터미널1: E4B+MTP, 포트 8000
docker compose up -d es                # 터미널2: Elasticsearch + nori

# 배선만 점검 — 모델·ES 불필요
python -m eval.run_regression --goldenset eval/goldenset_sample.jsonl --dry-run

# 실제 평가
python -m eval.run_regression \
    --goldenset eval/goldenset_sample.jsonl \
    --out eval/reports/latest.json
```

이번 결과를 베이스라인으로 굳히려면:

```bash
cp eval/reports/latest.json eval/reports/baseline.json
```

이후 변경분은 베이스라인과 비교해 회귀를 감지한다(핵심 지표 하락 시 exit 1):

```bash
python -m eval.run_regression \
    --goldenset eval/goldenset_sample.jsonl \
    --baseline eval/reports/baseline.json
```

회귀 임계치는 `run_regression.py`의 `REGRESSION_THRESHOLDS`에 있다 —
`Recall@5` −0.05, `MRR` −0.05, `faithfulness` −0.3, `correctness` −0.3.

> 골든셋을 새로 만들 때는 `gold_doc_ids`를 반드시 채운다. 비어 있으면
> `retrieval_eval.load_goldenset`이 해당 문항을 검색 평가에서 제외해
> Recall@5·MRR·nDCG가 항상 0으로 집계된다.
> 청크 ID는 `retrieval/indexer.py`의 `chunk()`가 만드는
> `sha1("{파일명}:{시작토큰}:{청크본문}")[:16]` 이다.

## 2. 일관성 검증

```bash
python -m eval.run_consistency --repeat 3
python -m eval.run_consistency --pairs eval/consistency_pairs_sample.jsonl --repeat 5
```

판정 기준:

- **invariant 페어**(c001, c002): 표현이 달라도 두 답이 같은 정답으로 수렴 → PASS
- **discriminative 페어**(c003): 중의적인 두 질문이 서로 다른 올바른 답 → PASS
- **route 안정성**: 같은 질문을 N회 반복했을 때 실행 경로가 일관되는지

기대 결과:

- c001 / c002: 두 답이 각각 같은 값(60 / nori)으로 수렴 → PASS
- c003: a는 RRF 융합 방식, b는 dense_vector 유사도 메트릭으로 갈려야 PASS

## 3. ADR

`docs/adr/0004-retrieval-k-and-verify-hardening.md`의 §6 후속 체크리스트는 비어 있다.
위 두 평가 결과 수치를 채워 넣으면 "측정 → 결정 → 검증" 사이클이 완결된다.

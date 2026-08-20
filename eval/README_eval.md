# 평가 산출물 사용 안내

이 디렉토리의 파일들을 워크스테이션(`/home/mozi/gemma-rag-harness`)의
`eval/`, `docs/adr/`에 복사해 사용하세요.

## 파일 목록

| 파일 | 용도 |
|---|---|
| `golden_50.jsonl` | 50문항 확장 골든셋(기존 15문항 형식 계승) |
| `consistency_pairs.jsonl` | 일관성 검증 페어 10쌍(중의성 구분 5 + 표현 불변 5) |
| `run_consistency.py` | 일관성 검증 러너 |
| `../docs/adr/ADR-0001-*.md` | BASE_K↑ + VERIFY 강화 결정 기록 |

## 1. 골든셋 재평가 (50문항)

기존 평가 스크립트에 골든셋 경로만 바꿔 넣으면 됩니다.

```bash
# 서빙·ES 기동 (표준 순서)
bash serving/vllm_launch.sh           # 터미널1: E4B+MTP, 포트 8000
docker start es                        # 터미널2

# 재평가 — 기존 평가 진입점에 50문항 골든셋 지정
python -m eval.run --golden eval/golden_50.jsonl --out eval/reports/latest_50.json

# baseline과 비교 (baseline.json이 있으면 자동 비교)
# correctness가 4.533(15문항 baseline) 대비 올랐는지 확인
```

> 15문항 baseline과 50문항은 문항 수가 다르므로, 엄밀히는 50문항으로 새
> baseline을 한 번 찍고(개선 전 코드로) 그 다음 개선 후를 비교하는 것이
> 가장 정확합니다. 이미 개선이 적용된 상태라면, 지금 50문항 결과를 새
> baseline(`baseline_50.json`)으로 저장해두고 이후 변경과 비교하세요.

```bash
cp eval/reports/latest_50.json eval/reports/baseline_50.json
```

## 2. 일관성 검증

```bash
# run_consistency.py 상단의 import 두 줄을 프로젝트에 맞게 확인:
#   from harness.graph import run_query        # query -> {answer, route, retry_count, ...}
#   from eval.judge import call_judge_model    # prompt -> str(JSON)

python -m eval.run_consistency --repeat 3
```

판정 기준:
- **invariant 페어**(c002, c003, c005, c007, c009): 두 답이 같은 정답으로
  수렴 → PASS. 개선 전 불안정성이 나던 지점.
- **discriminative 페어**(c001, c004, c006, c008, c010): 두 답이 서로 다른
  올바른 답 → PASS. 중의성을 구분하는지.
- **route 안정성**: 같은 질문 N회 반복 시 실행 경로가 일관되는지.

기대 결과(개선이 제대로 적용됐다면):
- c001: a=certification(GS인증), b=authentication(인증방식) → 서로 달라 PASS
- c002: 둘 다 GS인증으로 수렴 → PASS (개선 전엔 여기서 갈렸음)

## 3. ADR

`docs/adr/ADR-0001-retrieval-k-and-verify-hardening.md`는 위 두 평가 결과를
첨부하는 자리(§6 후속 체크리스트)를 비워뒀습니다. 재평가·일관성 결과 수치를
채워 넣으면 "측정 → 결정 → 검증" 사이클이 완결된 포트폴리오 증빙이 됩니다.

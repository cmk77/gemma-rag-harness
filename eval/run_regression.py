"""
gemma-rag-harness / eval/run_regression.py

전체 평가 파이프라인을 골든셋에 돌려 리포트를 만들고,
이전 베이스라인 대비 점수 하락(회귀)을 감지한다. CI에서 호출.

흐름:
  골든셋 → [각 질문마다 하네스 graph 실행] → 검색지표 + Judge지표 집계
        → JSON 리포트 저장 → 이전 리포트와 비교 → 회귀 시 exit 1

CI 연동(.github/workflows/ci.yml):
  - run: python -m eval.run_regression --baseline eval/reports/baseline.json
  회귀 감지 시 비정상 종료 → PR 머지 차단.

오프라인/모델 미연결 시:
  --dry-run 으로 파이프라인 배선만 점검(가짜 결과로 리포트 형식 확인).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 회귀 판정 임계치: 핵심 지표가 이 폭 이상 떨어지면 실패 처리
REGRESSION_THRESHOLDS = {
    "Recall@5": 0.05,       # 검색 회수율 5%p 이상 하락 → 회귀
    "MRR": 0.05,
    "faithfulness": 0.3,    # Judge 점수(1~5) 0.3 이상 하락 → 회귀
    "correctness": 0.3,
}


def run_pipeline(goldenset: list[dict], dry_run: bool = False) -> dict:
    """
    각 골든셋 질문에 대해 하네스를 실행하고 검색·생성 지표를 모은다.
    dry_run=True면 graph/judge를 호출하지 않고 가짜 결과로 형식만 만든다.
    """
    from eval.retrieval_eval import evaluate_retrieval

    if dry_run:
        # 배선 점검용 가짜 결과
        retrieval = {"aggregate": {"Recall@5": 0.0, "MRR": 0.0, "nDCG@5": 0.0,
                                   "Hit@5": 0.0, "n_queries": len(goldenset)}}
        judge_agg = {"n_total": len(goldenset), "n_parsed": 0,
                     "parse_fail_rate": 1.0, "faithfulness": None,
                     "correctness": None, "relevance": None}
        return {"retrieval": retrieval["aggregate"], "judge": judge_agg}

    # ── 실제 실행 (모델·ES 연결 필요) ──────────────────────
    from eval.judge import aggregate_judgments, judge_answer
    from harness.graph import build_graph

    app = build_graph()

    # 검색 평가용: 질문 → 검색된 doc_id 리스트.
    # 하네스 실행 결과의 _doc_pool에서 id를 뽑는다.
    def search_fn(question: str) -> list[str]:
        state = app.invoke({"query": question})
        pool = state.get("_doc_pool", [])
        return [d.id for d in pool]

    search_targets = [r for r in goldenset if r.get("gold_doc_ids")]
    retrieval = evaluate_retrieval(search_fn, search_targets, k=5)

    # 생성 품질 평가(Judge)
    judgments = []
    for row in goldenset:
        state = app.invoke({"query": row["question"]})
        answer = state.get("answer", "")
        ctx = state.get("docs", [])
        judgments.append(judge_answer(
            row["question"], answer, row.get("answer", ""), ctx))

    return {
        "retrieval": retrieval["aggregate"],
        "judge": aggregate_judgments(judgments),
    }


def detect_regression(current: dict, baseline: dict) -> list[str]:
    """베이스라인 대비 임계치 초과 하락을 찾아 사유 리스트로 반환."""
    failures = []
    flat_cur = {**current.get("retrieval", {}), **current.get("judge", {})}
    flat_base = {**baseline.get("retrieval", {}), **baseline.get("judge", {})}

    for metric, max_drop in REGRESSION_THRESHOLDS.items():
        cur, base = flat_cur.get(metric), flat_base.get(metric)
        if cur is None or base is None:
            continue
        drop = base - cur
        if drop > max_drop:
            failures.append(
                f"{metric}: {base:.3f} → {cur:.3f} (하락 {drop:.3f} > 허용 {max_drop})")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goldenset", default="eval/goldenset.jsonl")
    ap.add_argument("--baseline", default=None, help="비교할 이전 리포트 JSON")
    ap.add_argument("--out", default="eval/reports/latest.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    goldenset = [json.loads(line) for line in
                 Path(args.goldenset).read_text(encoding="utf-8").splitlines() if line.strip()]

    print(f"[regression] 골든셋 {len(goldenset)}건 실행"
          f"{' (dry-run)' if args.dry_run else ''}...")
    t0 = time.time()
    result = run_pipeline(goldenset, dry_run=args.dry_run)
    result["elapsed_sec"] = round(time.time() - t0, 2)
    result["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    # 리포트 저장
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[regression] 리포트 저장: {out_path}")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 회귀 판정
    if args.baseline and Path(args.baseline).exists():
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        failures = detect_regression(result, baseline)
        if failures:
            print("\n[regression] ❌ 회귀 감지:")
            for f in failures:
                print(f"  - {f}")
            return 1
        print("\n[regression] ✓ 회귀 없음 — 모든 핵심 지표가 허용 범위 내")
    else:
        print("\n[regression] (베이스라인 없음 — 비교 생략, 이번 결과를 베이스라인으로 쓸 수 있음)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
eval/run_consistency.py

일관성 검증 러너.

consistency_pairs.jsonl의 각 페어를 하네스로 실행하고, LLM-as-Judge로
두 답변을 판정한다.

- invariant 페어: 두 답변이 같은 정답으로 수렴하면 PASS (표현 변주 강건성)
- discriminative 페어: 두 답변이 서로 다른 올바른 답이면 PASS (중의성 구분)

개선(BASE_K↑ + VERIFY 강화)의 효과를 수치로 검증하는 것이 목적이다.
같은 페어를 N회 반복 실행해 답변의 안정성(재현성)도 함께 측정한다.

사용법:
    python -m eval.run_consistency                    # 기본 3회 반복
    python -m eval.run_consistency --repeat 5         # 5회 반복
    python -m eval.run_consistency --pairs eval/consistency_pairs.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# 하네스 진입점 — 프로젝트 구조에 맞게 조정.
# 채널의 scripts.ask가 노출하는 함수 시그니처에 맞춰 import 하세요.
# 예: from harness.graph import run_query  (query -> dict{answer, route, docs, retry_count})
try:
    from harness.graph import run_query  # type: ignore
except Exception:  # noqa: BLE001
    run_query = None  # 실행 환경(워크스테이션)에서만 import 되면 됨


JUDGE_PROMPT = """당신은 RAG 답변 정합성 판정관입니다.

[질문 A] {qa}
[답변 A] {aa}
[질문 A 기대 정답] {ga}

[질문 B] {qb}
[답변 B] {ab}
[질문 B 기대 정답] {gb}

[페어 유형] {ptype}
[판정 기준] {check}

다음을 JSON으로만 출력하세요(설명·마크다운 금지):
{{
  "a_correct": true|false,        // 답변 A가 기대 정답에 부합하는가
  "b_correct": true|false,        // 답변 B가 기대 정답에 부합하는가
  "relation_ok": true|false,      // 페어 유형 기준을 충족하는가
                                  //   invariant: 두 답이 같은 정답으로 수렴 → true
                                  //   discriminative: 두 답이 서로 다른 올바른 답 → true
  "reason": "한 문장 근거"
}}"""


def judge(pair: dict, ans_a: str, ans_b: str, judge_fn) -> dict:
    prompt = JUDGE_PROMPT.format(
        qa=pair["question_a"], aa=ans_a, ga=pair["gold_a"],
        qb=pair["question_b"], ab=ans_b, gb=pair["gold_b"],
        ptype=pair["type"], check=pair["check"],
    )
    raw = judge_fn(prompt)
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="eval/consistency_pairs.jsonl")
    ap.add_argument("--repeat", type=int, default=3,
                    help="각 질문을 몇 번 반복 실행해 안정성을 볼지")
    ap.add_argument("--out", default="eval/reports/consistency_latest.json")
    args = ap.parse_args()

    if run_query is None:
        print("ERROR: harness.graph.run_query 를 import 하지 못했습니다.")
        print("       워크스테이션(/home/mozi/gemma-rag-harness)에서 실행하고,")
        print("       import 경로를 프로젝트에 맞게 수정하세요.")
        return 1

    # judge_fn: 채널의 LLM-as-Judge 호출부를 재사용하세요.
    # 여기서는 하네스의 GENERATE와 같은 모델 백엔드를 직접 부르는 것을 가정.
    from eval.judge import call_judge_model as judge_fn  # type: ignore

    pairs = [json.loads(l) for l in open(args.pairs, encoding="utf-8")]
    results = []
    summary = defaultdict(lambda: {"pass": 0, "fail": 0})
    stability = []  # 반복 실행 시 답변이 일관됐는지

    for p in pairs:
        # 안정성: 각 질문을 repeat회 실행해 route/답변 요지가 일관되는지
        routes_a, routes_b, answers_a, answers_b = [], [], [], []
        for _ in range(args.repeat):
            ra = run_query(p["question_a"])
            rb = run_query(p["question_b"])
            routes_a.append(ra.get("route"))
            routes_b.append(rb.get("route"))
            answers_a.append(ra.get("answer", ""))
            answers_b.append(rb.get("answer", ""))

        # 마지막 실행 답변으로 정합성 판정
        verdict = judge(p, answers_a[-1], answers_b[-1], judge_fn)
        passed = verdict["a_correct"] and verdict["b_correct"] and verdict["relation_ok"]

        route_stable_a = len(set(routes_a)) == 1
        route_stable_b = len(set(routes_b)) == 1
        stability.append(route_stable_a and route_stable_b)

        summary[p["type"]]["pass" if passed else "fail"] += 1
        results.append({
            "pair_id": p["pair_id"],
            "type": p["type"],
            "topic": p["topic"],
            "passed": passed,
            "verdict": verdict,
            "route_stable": {"a": route_stable_a, "b": route_stable_b},
            "routes": {"a": routes_a, "b": routes_b},
            "answers": {"a": answers_a[-1], "b": answers_b[-1]},
        })
        mark = "PASS" if passed else "FAIL"
        stab = "안정" if (route_stable_a and route_stable_b) else "불안정"
        print(f"[{mark}] {p['pair_id']} ({p['type']:14s}) route:{stab}  {p['topic']}")

    total = len(results)
    n_pass = sum(r["passed"] for r in results)
    n_stable = sum(stability)
    print("\n=== 요약 ===")
    for t, s in summary.items():
        tot = s["pass"] + s["fail"]
        print(f"  {t:14s}: {s['pass']}/{tot} pass")
    print(f"  전체 정합성 : {n_pass}/{total} ({100*n_pass/total:.1f}%)")
    print(f"  route 안정성: {n_stable}/{total} ({100*n_stable/total:.1f}%)")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({
        "summary": {
            "total": total, "pass": n_pass,
            "pass_rate": n_pass / total,
            "route_stable": n_stable, "stable_rate": n_stable / total,
            "by_type": dict(summary),
        },
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n리포트 저장: {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

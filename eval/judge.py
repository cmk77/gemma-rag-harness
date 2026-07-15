"""
gemma-rag-harness / eval/judge.py

LLM-as-Judge: 생성된 답변을 Gemma 4 27B가 3축으로 채점한다.
검색 평가(retrieval_eval)가 '근거를 잘 가져왔나'를 본다면,
여기서는 '가져온 근거로 좋은 답을 썼나'를 본다.

3축 (각 1~5점):
  - faithfulness : 답변이 근거에 충실한가 (환각 없는가) ← RAG 핵심 지표
  - correctness  : 답변이 정답(gold answer)과 일치하는가
  - relevance    : 답변이 질문에 적절히 답하는가

함정: Judge 출력은 자유 텍스트라 파싱이 깨지기 쉽다.
  → JSON만 출력하도록 강제하고, 코드펜스/잡텍스트를 견고하게 벗겨낸 뒤 파싱.
  → 파싱 실패 시 0점이 아니라 None으로 표시해 '측정 실패'와 '낮은 점수'를 구분.

사용:
  from eval.judge import judge_answer
  result = judge_answer(question, answer, gold_answer, context_docs)
"""

from __future__ import annotations

import json
import re

from models.backends import ask

# 채점은 결정적이어야 재현 가능하므로 temperature=0.0 (ask 기본값).
# 모델·엔드포인트는 backends가 .env(LLM_MODEL 등)를 따라 관리한다.

JUDGE_PROMPT = """너는 RAG 답변 채점자다. 아래를 보고 세 항목을 각각 1~5점으로 채점하라.

- faithfulness: 답변이 [근거]에 의해 뒷받침되는가. 근거에 없는 주장(환각)이 있으면 낮게.
- correctness: 답변이 [정답]과 사실적으로 일치하는가.
- relevance: 답변이 [질문]에 적절히 답하는가.

채점 기준: 5=완벽, 4=경미한 결함, 3=부분적, 2=상당한 결함, 1=실패.

반드시 아래 JSON 형식만 출력하라. 다른 텍스트·설명·코드펜스 금지:
{"faithfulness": <1-5>, "correctness": <1-5>, "relevance": <1-5>, "reason": "<한 문장>"}"""


def _extract_json(text: str) -> dict | None:
    """모델 출력에서 JSON 객체를 견고하게 추출."""
    # 1) 코드펜스 제거
    text = re.sub(r"```(?:json)?", "", text).strip()
    # 2) 첫 '{' ~ 마지막 '}' 구간만 취함 (앞뒤 잡텍스트 방어)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _clamp(v, lo=1, hi=5):
    try:
        return max(lo, min(hi, int(round(float(v)))))
    except (TypeError, ValueError):
        return None


def judge_answer(
    question: str,
    answer: str,
    gold_answer: str,
    context_docs: list[str],
) -> dict:
    """
    반환: {faithfulness, correctness, relevance, reason, parsed(bool)}
    점수가 None이면 파싱 실패(측정 불가) — 집계에서 제외해야 한다.
    """
    ctx = "\n".join(f"[{i+1}] {d}" for i, d in enumerate(context_docs)) or "(근거 없음)"
    user = (
        f"[질문]\n{question}\n\n"
        f"[근거]\n{ctx}\n\n"
        f"[정답]\n{gold_answer}\n\n"
        f"[답변]\n{answer}"
    )
    raw = ask(JUDGE_PROMPT, user, temperature=0.0)
    data = _extract_json(raw)

    if data is None:
        return {"faithfulness": None, "correctness": None, "relevance": None,
                "reason": "JUDGE_PARSE_FAILED", "parsed": False, "raw": raw[:200]}

    return {
        "faithfulness": _clamp(data.get("faithfulness")),
        "correctness": _clamp(data.get("correctness")),
        "relevance": _clamp(data.get("relevance")),
        "reason": str(data.get("reason", ""))[:300],
        "parsed": True,
    }


def aggregate_judgments(judgments: list[dict]) -> dict:
    """파싱 성공한 건만 평균. 파싱 실패율도 함께 보고."""
    axes = ["faithfulness", "correctness", "relevance"]
    valid = [j for j in judgments if j.get("parsed")]
    out: dict = {"n_total": len(judgments), "n_parsed": len(valid)}
    out["parse_fail_rate"] = round(1 - len(valid) / (len(judgments) or 1), 4)
    for ax in axes:
        vals = [j[ax] for j in valid if j.get(ax) is not None]
        out[ax] = round(sum(vals) / len(vals), 3) if vals else None
    return out


if __name__ == "__main__":
    # 파서 단위 검증 (모델 호출 없이)
    samples = [
        '{"faithfulness": 5, "correctness": 4, "relevance": 5, "reason": "정확"}',
        '```json\n{"faithfulness": 3, "correctness": 2, "relevance": 4, "reason": "부분적"}\n```',
        '채점 결과는 다음과 같습니다: {"faithfulness": 4, "correctness": 4, "relevance": 4, "reason": "양호"} 입니다.',
        '판정 불가 (형식 깨짐)',
    ]
    for s in samples:
        print(_extract_json(s))

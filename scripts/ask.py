"""
scripts/ask.py

하네스에 질문을 던지고 답변·근거·실행 경로를 보여주는 스크립트.

사용:
  python -m scripts.ask                          # 대화형 모드
  python -m scripts.ask "RRF의 rank_constant는?"  # 단일 질문
  python -m scripts.ask --quiet "질문"            # 실행 경로 로그 없이 답변만

전제:
  - vLLM 서빙이 떠 있어야 함 (localhost:8000)
  - Elasticsearch가 떠 있고 코퍼스가 색인돼 있어야 함
    (python -m scripts.index_corpus 먼저 실행)

실행 경로 로그:
  하네스가 ROUTER → RETRIEVE → GENERATE → VERIFY 노드를 지날 때마다
  각 단계를 화면에 출력한다. 재검색이 일어나면 RETRIEVE가 여러 번 찍힌다.
  --quiet 를 주면 이 로그를 끄고 최종 답변만 본다.
"""

from __future__ import annotations

import logging
import sys


def setup_logging(quiet: bool) -> None:
    """하네스 실행 경로 로그를 화면에 보이도록 설정."""
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="  %(message)s",   # 앞에 들여쓰기만, 시각/레벨 생략해 깔끔하게
        stream=sys.stdout,
    )
    # 시끄러운 외부 라이브러리 로그는 억제 (하네스 로그만 보이게)
    for noisy in ("httpx", "urllib3", "elasticsearch", "sentence_transformers", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def ask_one(app, query: str, quiet: bool) -> None:
    print("\n" + "=" * 60)
    print(f"질문: {query}")
    if not quiet:
        print("-" * 60)
        print("[실행 경로]")
    # invoke 하는 동안 각 노드가 logger.info로 경로를 출력한다
    result = app.invoke({"query": query})

    print("-" * 60)
    print(f"답변: {result.get('answer', '(답변 없음)')}")
    print("-" * 60)

    # 종료 요약
    stop = result.get("stop_reason", "n/a")
    stop_kr = {
        "verified": "검증 통과",
        "max_retries": "재검색 상한 도달",
        "no_new_docs": "새 근거 없어 중단",
    }.get(stop, stop)
    print(f"종료: {stop_kr}"
          f" | 재검색 {result.get('retry_count', 0)}회"
          f" | 근거 {len(result.get('docs', []))}개")

    # 근거 문서 미리보기
    docs = result.get("docs", [])
    if docs:
        print("근거:")
        for i, d in enumerate(docs[:3], 1):
            preview = d[:80].replace("\n", " ")
            print(f"  [{i}] {preview}...")
    print("=" * 60)


def main() -> None:
    args = sys.argv[1:]
    quiet = "--quiet" in args
    args = [a for a in args if a != "--quiet"]

    setup_logging(quiet)

    print("하네스 초기화 중...")
    from harness.graph import build_graph
    app = build_graph()

    # 단일 질문 모드
    if args:
        ask_one(app, " ".join(args), quiet)
        return

    # 대화형 모드
    print("질문을 입력하세요 (종료: quit 또는 Ctrl-C)\n")
    try:
        while True:
            query = input("질문> ").strip()
            if query.lower() in {"quit", "exit", "q"}:
                break
            if query:
                ask_one(app, query, quiet)
    except (KeyboardInterrupt, EOFError):
        print("\n종료합니다.")


if __name__ == "__main__":
    main()

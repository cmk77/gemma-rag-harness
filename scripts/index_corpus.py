"""
scripts/index_corpus.py

corpus/ 디렉토리의 문서(.md, .txt)를 읽어 Elasticsearch에 색인한다.

사용:
  python -m scripts.index_corpus               # corpus/ 전체 색인
  python -m scripts.index_corpus --dir mydocs  # 다른 디렉토리 지정

전제: Elasticsearch가 ES_URL(기본 http://localhost:9200)에 떠 있어야 함.
      첫 실행 시 bge-m3 임베딩 모델(약 2GB)을 자동 다운로드한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from retrieval.indexer import index_corpus


def load_corpus(directory: str) -> dict[str, str]:
    """디렉토리의 .md/.txt 파일을 {파일명: 본문} 으로 읽는다."""
    corpus: dict[str, str] = {}
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"코퍼스 디렉토리 없음: {root.resolve()}")
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in {".md", ".txt"}:
            corpus[path.name] = path.read_text(encoding="utf-8")
    return corpus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="corpus", help="색인할 문서 디렉토리")
    args = ap.parse_args()

    corpus = load_corpus(args.dir)
    if not corpus:
        print(f"[index] {args.dir} 에서 색인할 문서(.md/.txt)를 찾지 못함")
        return

    print(f"[index] {len(corpus)}개 문서 색인 시작...")
    for name in corpus:
        print(f"  - {name}")

    n_chunks = index_corpus(corpus)
    print(f"[index] 완료: {n_chunks}개 청크 색인됨")


if __name__ == "__main__":
    main()

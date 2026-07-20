"""
scripts/extract_docs.py

원본 문서(PDF/HTML)를 마크다운으로 추출해 corpus/ 에 저장한다.
색인(index_corpus)의 전처리 단계.

  raw/pdf/*.pdf   ──pymupdf4llm──▶ corpus/<name>.md  (표·레이아웃 보존, 필요시 OCR)
  raw/html/*.html ──trafilatura──▶ corpus/<name>.md  (본문만 정제, 잡음 제거)

사용:
  python -m scripts.extract_docs                 # raw/ 전체 추출
  python -m scripts.extract_docs --pdf-only      # PDF만
  python -m scripts.extract_docs --html-only     # HTML만
  python -m scripts.extract_docs --preview 파일  # 한 파일만 추출해 화면에 미리보기

의존: pip install pymupdf4llm trafilatura
"""

from __future__ import annotations

import argparse
from pathlib import Path

RAW_PDF = Path("raw/pdf")
RAW_HTML = Path("raw/html")
OUT_DIR = Path("corpus")

# 추출 결과가 이 길이 미만이면 "추출 실패 의심"으로 경고 (빈 껍데기 방지)
MIN_CHARS = 50


def extract_pdf(path: Path) -> str:
    """PDF → 마크다운. 표를 마크다운 표로, 이미지 페이지는 자동 OCR."""
    import pymupdf4llm
    # header/footer=False: 페이지마다 반복되는 로고·페이지번호 제거
    return pymupdf4llm.to_markdown(str(path), show_progress=False)


def extract_html(path: Path) -> str:
    """HTML → 마크다운. 본문만 추출(네비·광고·푸터 제거)."""
    import trafilatura
    html = path.read_text(encoding="utf-8", errors="ignore")
    # output_format='markdown': 제목·목록 구조를 마크다운으로 보존
    result = trafilatura.extract(
        html,
        output_format="markdown",
        include_tables=True,
        include_links=False,
        favor_recall=True,       # 본문을 놓치지 않도록 (약간의 잡음 감수)
    )
    return result or ""


def _slug(name: str) -> str:
    """파일명을 안전한 슬러그로. 한글은 유지, 공백·특수문자만 정리."""
    stem = Path(name).stem
    return stem.replace(" ", "_").replace("/", "_")


def process(kind: str, src_dir: Path, pattern: str, extractor) -> None:
    """한 종류(pdf/html)의 파일들을 추출해 corpus/ 에 저장."""
    files = sorted(src_dir.glob(pattern))
    if not files:
        print(f"[{kind}] {src_dir}/{pattern} 에 파일 없음 (건너뜀)")
        return

    OUT_DIR.mkdir(exist_ok=True)
    print(f"[{kind}] {len(files)}개 파일 추출 시작...")

    ok, warn = 0, 0
    for f in files:
        try:
            text = extractor(f)
        except Exception as e:
            print(f"  ✗ {f.name}: 추출 실패 ({type(e).__name__}: {e})")
            warn += 1
            continue

        n = len(text.strip())
        out_path = OUT_DIR / f"{_slug(f.name)}.md"
        out_path.write_text(text, encoding="utf-8")

        if n < MIN_CHARS:
            print(f"  ⚠ {f.name}: 추출 텍스트 매우 짧음 ({n}자) — 이미지 위주 문서일 수 있음")
            warn += 1
        else:
            print(f"  ✓ {f.name} → {out_path.name} ({n:,}자)")
            ok += 1

    print(f"[{kind}] 완료: 정상 {ok}개, 확인필요 {warn}개")


def preview(path_str: str) -> None:
    """한 파일만 추출해 앞부분을 화면에 출력(색인 전 품질 확인용)."""
    path = Path(path_str)
    if not path.exists():
        print(f"파일 없음: {path}")
        return
    if path.suffix.lower() == ".pdf":
        text = extract_pdf(path)
    elif path.suffix.lower() in {".html", ".htm"}:
        text = extract_html(path)
    else:
        print(f"지원 안 함: {path.suffix}")
        return

    print(f"=== {path.name} 추출 결과 ({len(text):,}자) ===\n")
    print(text[:2000])
    if len(text) > 2000:
        print(f"\n... (이하 {len(text)-2000:,}자 생략)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-only", action="store_true")
    ap.add_argument("--html-only", action="store_true")
    ap.add_argument("--preview", metavar="FILE", help="한 파일만 추출해 미리보기")
    args = ap.parse_args()

    if args.preview:
        preview(args.preview)
        return

    do_pdf = not args.html_only
    do_html = not args.pdf_only

    if do_pdf:
        process("PDF", RAW_PDF, "*.pdf", extract_pdf)
    if do_html:
        process("HTML", RAW_HTML, "*.html", extract_html)

    print("\n추출 완료. corpus/ 내용을 확인한 뒤 색인하세요:")
    print("  python -m scripts.index_corpus")


if __name__ == "__main__":
    main()

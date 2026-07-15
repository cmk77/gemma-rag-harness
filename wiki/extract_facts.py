"""
wiki/extract_facts.py — MAP 단계: 문서 → Fact JSONL (병렬·증분·재개 가능).

수백만 문서 전제의 설계 원칙:
  1. 문서별 완전 독립 → ThreadPool 병렬(HTTP 호출 병렬화). 진짜 대규모에선
     WORKERS를 늘리고 vLLM 레플리카/큐(Ray·Celery)로 수평 확장 — 이 파일의
     map 함수는 상태가 없어서 그대로 옮겨진다.
  2. content-hash 레지스트리(state/processed.jsonl) → 재실행 시 내용이 같은
     문서는 스킵. 파이프라인이 죽어도 이어서 재개. 문서가 갱신되면 새 해시로
     재추출되고, 옛 해시의 사실은 build 단계에서 자동 무효화.
  3. LLM 출력은 신뢰하지 않는다 → JSON 파싱 실패는 state/errors.jsonl 에
     적재하고 계속 진행(한 문서가 전체를 못 멈춤). 나중에 재처리 대상.
  4. 산출은 append-only JSONL. 대규모에선 parquet/DB로 바꾸되 레코드 스키마
     (wiki/schema.Fact)는 동일.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from wiki.schema import Fact, doc_hash, valid_section

STATE_DIR = Path("wiki_state")
FACTS_PATH = STATE_DIR / "facts.jsonl"
REGISTRY_PATH = STATE_DIR / "processed.jsonl"   # {"source":..., "doc_hash":...}
ERRORS_PATH = STATE_DIR / "errors.jsonl"

CHUNK_CHARS = 2400        # LLM 1콜당 입력 조각 크기 (문단 경계 우선)
PROMPT_NAME = "wiki_extract"


# ── 프롬프트 로딩 (harness.prompts 재사용) ─────────────────
def _load_prompt() -> str:
    from harness.prompts import load_prompt
    return load_prompt(PROMPT_NAME)


# ── 조각내기: 문단 경계 우선, 길면 강제 분할 ────────────────
def split_chunks(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                chunks.append(buf)
            while len(p) > max_chars:            # 한 문단이 과대하면 강제 분할
                chunks.append(p[:max_chars])
                p = p[max_chars:]
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


# ── LLM JSON 방어적 파싱 ───────────────────────────────────
def parse_facts_json(raw: str) -> list[dict]:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)   # 코드펜스 제거
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        i, j = s.find("{"), s.rfind("}")
        if i == -1 or j <= i:
            raise
        data = json.loads(s[i:j + 1])                # 앞뒤 잡담 제거 재시도
    facts = data.get("facts", [])
    return facts if isinstance(facts, list) else []


# ── 레지스트리 (증분 처리의 심장) ───────────────────────────
def load_registry() -> dict[str, str]:
    """source → 최신 doc_hash. build 단계도 이걸로 유효 사실을 판별한다."""
    reg: dict[str, str] = {}
    if REGISTRY_PATH.exists():
        for line in REGISTRY_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                reg[r["source"]] = r["doc_hash"]     # 뒤 레코드가 앞을 덮음
    return reg


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── 문서 1건 처리 (상태 없음 → 병렬·분산 안전) ──────────────
def extract_one(source: str, text: str, system_prompt: str, ask_fn) -> list[Fact]:
    h = doc_hash(text)
    facts: list[Fact] = []
    for chunk in split_chunks(text):
        raw = ask_fn(system_prompt, f"[문서 조각] (출처: {source})\n{chunk}",
                     max_tokens=1024, role="expand")
        try:
            rows = parse_facts_json(raw)
        except Exception as e:
            _append_jsonl(ERRORS_PATH, [{"source": source, "doc_hash": h,
                                         "error": str(e)[:200], "raw": raw[:500]}])
            continue
        for r in rows:
            ent, fct = str(r.get("entity", "")).strip(), str(r.get("fact", "")).strip()
            if ent and fct:
                facts.append(Fact(entity_raw=ent, section=valid_section(r.get("section")),
                                  fact=fct, source=source, doc_hash=h))
    return facts


def run_extract(corpus: dict[str, str], workers: int = 4,
                ask_fn=None) -> tuple[int, int, set[str]]:
    """
    corpus의 신규/변경 문서만 병렬 추출해 facts.jsonl에 append.
    반환: (처리 문서 수, 추출 사실 수, 영향받은 entity_raw 집합=dirty)
    """
    if ask_fn is None:
        from models.backends import ask as ask_fn  # noqa: PLW0127 — 지연 임포트
    system_prompt = _load_prompt()
    registry = load_registry()

    todo = {s: t for s, t in corpus.items() if registry.get(s) != doc_hash(t)}
    print(f"[wiki/extract] 대상 {len(todo)}개 (전체 {len(corpus)}개 중 "
          f"{len(corpus) - len(todo)}개는 변경 없음 → 스킵)")
    if not todo:
        return 0, 0, set()

    n_facts, dirty = 0, set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_one, s, t, system_prompt, ask_fn): (s, t)
                   for s, t in todo.items()}
        for fut in as_completed(futures):
            source, text = futures[fut]
            facts = fut.result()
            _append_jsonl(FACTS_PATH, [f.to_json() for f in facts])
            _append_jsonl(REGISTRY_PATH, [{"source": source, "doc_hash": doc_hash(text)}])
            n_facts += len(facts)
            dirty |= {f.entity_raw for f in facts}
            print(f"  ✓ {source}: 사실 {len(facts)}건")
    return len(todo), n_facts, dirty

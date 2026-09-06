"""
scripts/build_wiki.py — 원문 코퍼스를 "엔티티 중심 위키 페이지" 코퍼스로 재구성.

  corpus/*.md ──[A 추출]──▶ state/facts.jsonl ──[B 통합]──▶ ──[C 합성]──▶ corpus_wiki/*.md

왜 위키형인가:
  기계적 청킹은 서로 다른 주제(예: '취득한 인증' vs '지원하는 인증 방식')가
  한 청크에 섞여 검색 중의성을 만든다. 엔티티당 한 페이지 + 고정 섹션 스키마로
  재구성하면 섹션이 곧 청크 경계가 되어, 검색이 데이터 차원에서 정확해진다.

대규모(수천~수백만 문서) 전제의 설계:
  - map/reduce: A단계는 문서별 완전 독립(map) → 워커 수만 늘리면 수평 확장,
    파일 목록을 쪼개 여러 머신에서 돌려도 됨(JSONL append 병합만 하면 됨).
  - 증분(incremental): 문서 내용 sha1을 기록해 변경된 문서만 재처리.
    C단계도 엔티티별 사실 해시로 변경된 페이지만 재합성.
  - 재개(resumable): 모든 산출이 JSONL append + done 마커. 중단돼도 이어서 실행.
  - 견고성: LLM의 JSON 파싱 실패는 errors.jsonl에 기록하고 계속(파이프라인 불사).
  - 폭주 방어: 엔티티당 사실 상한(MAX_FACTS, 섹션 라운드로빈) — 수백만 문서에서
    한 엔티티에 사실 수만 개가 몰려 컨텍스트를 터뜨리는 것을 막는다.

사용 (vLLM 서빙이 떠 있어야 함):
  python -m scripts.build_wiki                          # corpus/ → corpus_wiki/
  python -m scripts.build_wiki --workers 8              # 병렬 LLM 호출
  python -m scripts.build_wiki --limit 100              # 앞 100개 문서만(파일럿)
  python -m scripts.build_wiki --rebuild                # 체크포인트 무시 전체 재빌드

효과 검증(A/B):
  ES_INDEX=gemma_rag_wiki python -m scripts.index_corpus --dir corpus_wiki
  ES_INDEX=gemma_rag_wiki python -m eval.run_regression --goldenset eval/goldenset_sample.jsonl
  → 기존 인덱스(gemma_rag) 점수와 비교.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from models.backends import ask

# ── 스키마: 섹션 집합 (인증 중의성 분리가 스키마에 박혀 있음) ──────────
SECTIONS = ["개요", "주요 기능", "취득 인증·수상", "지원 인증·규격", "도입·활용", "관계·연관"]

WINDOW_CHARS = 3000        # A단계: 문서를 이 크기 창으로 잘라 LLM에 투입
MAX_FACTS = 80             # 엔티티당 사실 상한(섹션 라운드로빈으로 선별)
EXTRACT_MAX_TOKENS = 2048   # 연혁·표 윈도우는 사실 수십 개 → 1024로는 JSON이 잘림(실측)
SYNTH_MAX_TOKENS = 3072   # 60+ facts 페이지가 1024로 잘려 뒤쪽 섹션(취득 인증 등) 유실(실측) → 여유 확보

EXTRACT_PROMPT = f"""너는 지식 추출기다. 주어진 문서 조각에서 '개체(entity)'와 '사실(fact)'을 추출하라.
반드시 JSON 배열만 출력하라(설명·코드펜스 금지). 각 원소:
{{"entity": "개체명", "type": "제품|회사|기술|기타", "section": "{'|'.join(SECTIONS)}", "fact": "한 문장 사실"}}

규칙:
- 문서에 없는 내용을 지어내지 마라.
- '취득 인증·수상'은 품질인증·수상 등 개체가 받은 자격, '지원 인증·규격'은 개체가
  기능으로 제공하는 인증 방식(OAuth2, JWT 등)·표준 규격이다. 절대 혼동하지 마라.
- 개체명은 문서 표기 그대로(예: API 게이트웨이, 검색 엔진).
- 표·연혁이 OCR로 깨져 뒤섞인 조각이라도, '개체 + 인증/수상 + 등급/연도'가
  식별되면 반드시 사실로 추출하라. 예: "API 게이트웨이 품질인증 1등급 취득"
  → entity "API 게이트웨이", section "취득 인증·수상", fact "품질인증 1등급을 취득했다".
- 인증·수상 사실의 entity는 인증 이름(예: "품질 인증")이 아니라 **그것을 받은
  제품/회사명**이다. 조각 안에 제품명이 안 보이면 [문서명]에서 추정하라.
- 연도·날짜(예: "2022년도")를 개체로 삼지 마라 — 사실 문장 안에 포함시켜라.
- 사실이 없으면 빈 배열 []을 출력하라."""

SYNTH_PROMPT = """너는 지식위키 편집자다. 아래 [사실 목록]만 사용해 개체의 위키 페이지를
마크다운으로 작성하라. 사실에 없는 내용은 절대 추가하지 마라.

형식(해당 사실이 있는 섹션만 포함, 순서 유지):
# <개체명>
## 개요
## 주요 기능
## 취득 인증·수상
## 지원 인증·규격
## 도입·활용
## 관계·연관

각 항목은 "- 사실 (출처: 파일명)" 형태의 불릿으로 쓰고, 중복은 합쳐라.
불릿은 간결한 한 문장으로 써라(장황한 부연 금지). 어떤 섹션도 빠뜨리지 마라 —
해당 사실이 있는 섹션은 반드시 모두 출력하라."""


# ── 유틸 ───────────────────────────────────────────────────────────
def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def _slug(name: str) -> str:
    s = re.sub(r"[\\/:*?\"<>|]", "", name).strip().replace(" ", "_")
    return s or "unknown"


def _parse_json_array(raw: str) -> list[dict]:
    """LLM 출력에서 JSON 배열을 견고하게 파싱.
    코드펜스·잡문 방어 + max_tokens로 배열이 중간에 잘린 경우
    마지막으로 완성된 객체까지 복구한다(사실 일부라도 건진다)."""
    raw = raw.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.M).strip()
    m = re.search(r"\[.*", raw, flags=re.S)      # 여는 대괄호부터 끝까지
    if not m:
        raise ValueError("JSON 배열 없음")
    frag = m.group(0)
    try:
        data = json.loads(frag)
    except json.JSONDecodeError:
        cut = frag.rfind("}")                     # 마지막 완성 객체까지 절단
        if cut == -1:
            raise
        data = json.loads(frag[:cut + 1] + "]")
    if not isinstance(data, list):
        raise ValueError("배열 아님")
    return data


class Jsonl:
    """스레드 안전 JSONL append 기록기(재개 가능 파이프라인의 기본 단위)."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out


# ── A단계: 문서별 사실 추출 (map — 완전 병렬·증분) ──────────────────
def stage_extract(src: Path, state: Path, workers: int, limit: int | None,
                  rebuild: bool) -> None:
    facts = Jsonl(state / "facts.jsonl")
    done = Jsonl(state / "done_extract.jsonl")
    errors = Jsonl(state / "errors.jsonl")
    done_map = {} if rebuild else {d["source"]: d["sha1"] for d in done.load()}

    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in {".md", ".txt"})
    if limit:
        files = files[:limit]

    todo = []
    for p in files:
        text = p.read_text(encoding="utf-8")
        h = _sha1(text)
        if done_map.get(p.name) == h:
            continue                      # 증분: 내용 동일 → 스킵
        todo.append((p.name, text, h))

    print(f"[A 추출] 대상 {len(todo)}개 / 전체 {len(files)}개 (스킵 {len(files)-len(todo)})")

    def _one(name: str, text: str, h: str) -> tuple[str, int]:
        n = 0
        for i in range(0, len(text), WINDOW_CHARS):
            window = text[i:i + WINDOW_CHARS]
            try:
                raw = ask(EXTRACT_PROMPT, f"[문서명] {name}\n\n{window}", role="expand",
                          max_tokens=EXTRACT_MAX_TOKENS)
                for item in _parse_json_array(raw):
                    ent = str(item.get("entity", "")).strip()
                    fact = str(item.get("fact", "")).strip()
                    sec = item.get("section", "기타")
                    if not ent or not fact:
                        continue
                    if sec not in SECTIONS:
                        sec = "개요"
                    facts.append({"entity": ent, "type": item.get("type", "기타"),
                                  "section": sec, "fact": fact, "source": name})
                    n += 1
            except Exception as e:  # 파싱/호출 실패 — 기록하고 계속
                errors.append({"stage": "extract", "source": name,
                               "offset": i, "error": str(e)[:200]})
        done.append({"source": name, "sha1": h})
        return name, n

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_one, *t) for t in todo]
            for fut in as_completed(futs):
                name, n = fut.result()
                print(f"  ✓ {name}: 사실 {n}개")


# ── B단계: 엔티티 통합 (reduce — 정규화·중복제거·상한) ─────────────
def _squash(s: str) -> str:
    """구두점·공백을 제거한 비교키. 'API - Gateway' / 'API-Gateway' /
    'API 게이트웨이(API Gateway: APIGW)' 같은 표기 변형을 흡수한다."""
    return re.sub(r"[\s\-:·().,\[\]/]+", "", s).lower()


def _canonical(name: str, aliases: dict[str, str]) -> str:
    key = re.sub(r"\s+", " ", name).strip()
    sk = _squash(key)
    for alias, canon in aliases.items():
        if sk == _squash(alias):
            return canon
    return key


def stage_resolve(state: Path, aliases: dict[str, str],
                  min_facts: int) -> dict[str, list[dict]]:
    rows = Jsonl(state / "facts.jsonl").load()
    grouped: dict[str, list[dict]] = {}
    seen: set[tuple[str, str]] = set()
    for r in rows:
        canon = _canonical(r["entity"], aliases)
        key = (canon, re.sub(r"\s+", "", r["fact"]).lower())
        if key in seen:                    # 사실 중복 제거(정규화 후 동일 문장)
            continue
        seen.add(key)
        grouped.setdefault(canon, []).append(r)

    # 폭주 방어: 섹션 라운드로빈으로 다양성 있게 MAX_FACTS 선별
    for ent, fs in grouped.items():
        if len(fs) > MAX_FACTS:
            by_sec: dict[str, list[dict]] = {}
            for f in fs:
                by_sec.setdefault(f["section"], []).append(f)
            picked, i = [], 0
            while len(picked) < MAX_FACTS:
                added = False
                for sec in SECTIONS:
                    bucket = by_sec.get(sec, [])
                    if i < len(bucket):
                        picked.append(bucket[i])
                        added = True
                        if len(picked) >= MAX_FACTS:
                            break
                if not added:
                    break
                i += 1
            grouped[ent] = picked

    grouped = {e: fs for e, fs in grouped.items() if len(fs) >= min_facts}
    print(f"[B 통합] 엔티티 {len(grouped)}개 (min_facts={min_facts} 필터 후)")
    return grouped


# ── C단계: 위키 페이지 합성 (map over entities — 증분) ─────────────
def stage_synthesize(grouped: dict[str, list[dict]], out: Path, state: Path,
                     workers: int, rebuild: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = Jsonl(state / "manifest.jsonl")
    errors = Jsonl(state / "errors.jsonl")
    prev = {} if rebuild else {m["entity"]: m["facts_sha1"] for m in manifest.load()}

    def _facts_hash(fs: list[dict]) -> str:
        return _sha1("\n".join(sorted(f["fact"] for f in fs)))

    todo = []
    for ent, fs in sorted(grouped.items()):
        h = _facts_hash(fs)
        if prev.get(ent) == h and (out / f"{_slug(ent)}.md").exists():
            continue                      # 증분: 사실 집합 동일 → 재합성 스킵
        todo.append((ent, fs, h))

    print(f"[C 합성] 대상 {len(todo)}개 / 전체 {len(grouped)}개 페이지")

    def _one(ent: str, fs: list[dict], h: str) -> str:
        lines = [f"- ({f['section']}) {f['fact']} (출처: {f['source']})" for f in fs]
        user = f"개체명: {ent}\n\n[사실 목록]\n" + "\n".join(lines)
        try:
            page = ask(SYNTH_PROMPT, user, role="generate",
                       max_tokens=SYNTH_MAX_TOKENS)
        except Exception as e:
            errors.append({"stage": "synth", "entity": ent, "error": str(e)[:200]})
            return f"✗ {ent}"
        path = out / f"{_slug(ent)}.md"
        path.write_text(page.strip() + "\n", encoding="utf-8")
        manifest.append({"entity": ent, "file": path.name,
                         "n_facts": len(fs), "facts_sha1": h})
        return f"✓ {ent} ({len(fs)} facts)"

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(_one, *t) for t in todo]):
                print(f"  {fut.result()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="corpus", help="원문 코퍼스 디렉토리")
    ap.add_argument("--out", default="corpus_wiki", help="위키 페이지 출력 디렉토리")
    ap.add_argument("--state", default="state/wiki", help="체크포인트 디렉토리")
    ap.add_argument("--workers", type=int, default=4, help="병렬 LLM 호출 수")
    ap.add_argument("--limit", type=int, default=None, help="앞 N개 문서만(파일럿)")
    ap.add_argument("--min-facts", type=int, default=2,
                    help="이 미만 사실의 엔티티는 페이지 생성 생략(노이즈 컷)")
    ap.add_argument("--aliases", default=None,
                    help="별칭 JSON 경로 {\"별칭\": \"정식명\"} (선택)")
    ap.add_argument("--rebuild", action="store_true", help="체크포인트 무시 전체 재빌드")
    args = ap.parse_args()

    # 별칭 사전은 코퍼스마다 다르므로 하드코딩하지 않는다.
    # 표기 흔들림이 있으면 --aliases 로 JSON을 넘긴다 (예시: aliases_example.json).
    aliases: dict[str, str] = {}
    if args.aliases:
        aliases.update(json.loads(Path(args.aliases).read_text(encoding="utf-8")))

    src, out, state = Path(args.src), Path(args.out), Path(args.state)

    stage_extract(src, state, args.workers, args.limit, args.rebuild)
    grouped = stage_resolve(state, aliases, args.min_facts)
    stage_synthesize(grouped, out, state, args.workers, args.rebuild)

    print("\n완료. 다음으로 별도 인덱스에 색인해 A/B 비교:")
    print(f"  ES_INDEX=gemma_rag_wiki python -m scripts.index_corpus --dir {out}")
    print("  ES_INDEX=gemma_rag_wiki python -m eval.run_regression --goldenset eval/goldenset_sample.jsonl")


if __name__ == "__main__":
    main()

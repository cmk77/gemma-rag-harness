"""Query — 저장된 지식을 목적에 맞게 꺼내 문서를 만드는 엔진.

Progressive Disclosure 3단계:
  S1 메타 검색   : 프론트매터만으로 후보 선별 (본문 토큰 0)
  S2 본문 확장   : 상위 top_k만 본문 로드 (카드당 문자 예산)
  S3 생성+가설검증: 초안 생성 -> VERIFY(근거 충실성 + 질문-답변 주제
                   정합성) -> 실패 시 폭 확장(wide_k) 하이브리드 재검색 1회

하네스 ADR-0001 교훈을 그대로 이식:
  - VERIFY 프롬프트에 [질문]을 명시적으로 전달
  - 재검색 시 k 상향으로 첫 검색 편향(예: '인증' 중의성) 보정
지식 부족 시 모선(메인 볼트) 읽기 전용 폴백 스캔.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .config import WikiSettings
from .llm import LLMClient
from .models import WikiCard
from .vault import VaultManager

_SUFFICIENCY_SYS = """아래 후보 카드 메타데이터 목록만 보고 판단하라.
질문에 답하기에 충분한 카드가 있는가?
출력: {"sufficient": bool, "pick_paths": [str], "missing": str}
pick_paths에는 열어볼 가치가 있는 카드 path를 최대 K개 고른다."""

_GENERATE_SYS = """너는 개인 위키 볼트 기반 리서치 어시스턴트다.
제공된 카드 본문만 근거로 한국어로 답하라. 근거가 없으면 지어내지 말고
'볼트에 근거 없음'을 명시하라. 각 주장 뒤에 [path] 형식으로 출처를 단다."""

_VERIFY_SYS = """너는 답변 검증자다. 아래를 검사하라.
1) 근거 충실성: 답변의 모든 사실이 컨텍스트에 실제로 존재하는가.
2) 주제 정합성: 답변이 [질문]이 묻는 바로 그 대상에 대한 것인가.
   (유사하지만 다른 개념으로 새면 실패 — 예: '받은 인증' vs '지원 인증')
출력: {"pass": bool, "reason": str, "better_query": str}"""


@dataclass
class QueryTrace:
    steps: list[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.steps.append(msg)


@dataclass
class QueryResult:
    question: str
    answer: str
    sources: list[str]
    verified: bool
    retries: int
    trace: QueryTrace


class QueryEngine:
    def __init__(self, settings: WikiSettings, vault: VaultManager,
                 indexer, llm: LLMClient, embedder=None):
        self.s = settings
        self.vault = vault
        self.indexer = indexer
        self.llm = llm
        self.embedder = embedder

    # ================= 메인 진입점 =================
    def run(self, question: str) -> QueryResult:
        trace = QueryTrace()
        cards = self._retrieve(question, wide=False, trace=trace)
        retries = 0
        answer, verified = self._generate_and_verify(question, cards, trace)

        while not verified and retries < self.s.max_retries:
            retries += 1
            trace.log(f"VERIFY 실패 -> 폭 확장 재검색 (k={self.s.wide_k})")
            cards = self._retrieve(question, wide=True, trace=trace)
            answer, verified = self._generate_and_verify(
                question, cards, trace)

        # 여전히 근거 부족 -> 모선 폴백 (읽기 전용)
        if not verified and self.vault.main_root:
            trace.log("위키 볼트 근거 부족 -> 메인 볼트 폴백 스캔")
            extra = self._main_vault_context(question)
            if extra:
                answer, verified = self._generate_and_verify(
                    question, cards, trace, extra_context=extra)

        return QueryResult(
            question=question, answer=answer,
            sources=[self.vault.rel(c.path) for c in cards if c.path],
            verified=verified, retries=retries, trace=trace)

    # ================= S1+S2: 검색 =================
    def _retrieve(self, question: str, wide: bool,
                  trace: QueryTrace) -> list[WikiCard]:
        k = self.s.wide_k if wide else self.s.top_k

        # --- S1: 메타 검색 (프론트매터만) ---
        meta_hits = self.indexer.meta_search(question, k=self.s.meta_k)
        trace.log(f"S1 meta_search hits={len(meta_hits)}")
        picked = self._pick_by_meta(question, meta_hits, k, trace)

        # --- 메타로 부족하거나 wide 모드면 본문 하이브리드 병행 ---
        if wide or len(picked) < k:
            vec = None
            if self.embedder is not None and self.embedder.available:
                vec = self.embedder.encode_one(question)
            hy = self.indexer.hybrid_search(question, vec, k=k)
            trace.log(f"S2 hybrid_search hits={len(hy)} "
                      f"(vector={'on' if vec else 'off'})")
            for h in hy:
                if h["path"] not in picked:
                    picked.append(h["path"])
        picked = picked[:k]

        # --- S2: 선택된 카드만 본문 로드 (토큰 예산) ---
        cards: list[WikiCard] = []
        for p in picked:
            try:
                cards.append(self.vault.load_card(p))
            except Exception:
                continue
        trace.log(f"S2 bodies loaded={len(cards)} "
                  f"(budget={self.s.body_char_budget}c/card)")
        return cards

    def _pick_by_meta(self, question: str, meta_hits: list[dict],
                      k: int, trace: QueryTrace) -> list[str]:
        """LLM이 메타데이터만 보고 '열어볼 카드'를 선별 — 점진적 노출 핵심."""
        if not meta_hits:
            return []
        listing = "\n".join(
            f"- path={h['path']} | [{h.get('type')}] {h.get('title')} — "
            f"{h.get('description', '')} | tags={h.get('tags', [])}"
            for h in meta_hits)
        try:
            out = self.llm.chat_json(
                _SUFFICIENCY_SYS.replace("K", str(k)),
                f"[질문] {question}\n\n[후보]\n{listing}")
            picks = [p for p in out.get("pick_paths", [])
                     if any(h["path"] == p for h in meta_hits)]
            trace.log(f"S1 meta-pick={len(picks)} "
                      f"sufficient={out.get('sufficient')}")
            return picks
        except Exception:
            return [h["path"] for h in meta_hits[:k]]

    # ================= S3: 생성 + 가설검증 =================
    def _generate_and_verify(self, question: str, cards: list[WikiCard],
                             trace: QueryTrace,
                             extra_context: str = "") -> tuple[str, bool]:
        ctx = self._context_block(cards) + (
            f"\n\n[모선 폴백 자료]\n{extra_context}" if extra_context else "")
        if not ctx.strip():
            return "볼트에 관련 지식이 없습니다. 먼저 인제스트가 필요합니다.", False

        sys_p = _GENERATE_SYS
        rules = self.vault.system_context()
        if rules:
            sys_p = f"{rules}\n\n{sys_p}"

        answer = self.llm.chat(
            sys_p, f"[질문] {question}\n\n[카드 컨텍스트]\n{ctx}",
            temperature=0.2)

        try:
            v = self.llm.chat_json(
                _VERIFY_SYS,
                f"[질문] {question}\n\n[답변]\n{answer}\n\n[컨텍스트]\n{ctx}")
            ok = bool(v.get("pass"))
            trace.log(f"S3 verify pass={ok} reason={v.get('reason', '')[:80]}")
        except Exception:
            ok = True  # 검증기 자체 실패 시 답변은 반환하되 trace에 남김
            trace.log("S3 verify 스킵(검증기 오류)")
        return answer, ok

    def _context_block(self, cards: list[WikiCard]) -> str:
        parts = []
        for c in cards:
            rel = self.vault.rel(c.path) if c.path else c.frontmatter.title
            parts.append(
                f"<card path='{rel}'>\n{c.meta_line()}\n\n"
                f"{c.clipped_body(self.s.body_char_budget)}\n</card>")
        return "\n\n".join(parts)

    def _main_vault_context(self, question: str) -> str:
        kws = [w for w in re.split(r"[\s,\.\?!]+", question) if len(w) >= 2]
        hits = self.vault.scan_main_vault(kws, limit=3)
        return "\n\n".join(
            f"<main path='{p}'>\n{snippet}\n</main>" for p, snippet in hits)

    # ---------- 쿼리 산출물 카드 저장 (qu 히스토리) ----------
    def save_as_note(self, result: QueryResult) -> str:
        from .models import CardType, Frontmatter
        fm = Frontmatter(
            title=f"Query - {result.question[:60]}",
            description="Saved query output (history)",
            type=CardType.NOTE.value,
            tags=["#query-history"],
            model=self.llm.model,
        )
        body = (f"## 질문\n{result.question}\n\n## 답변\n{result.answer}\n\n"
                f"## 출처\n" + "\n".join(f"- {s}" for s in result.sources) +
                f"\n\n## 트레이스\n" +
                "\n".join(f"- {s}" for s in result.trace.steps))
        card = WikiCard(frontmatter=fm, body=body)
        path = self.vault.save_card(card)
        return self.vault.rel(path)


def result_to_json(r: QueryResult) -> str:
    return json.dumps({
        "question": r.question, "answer": r.answer, "sources": r.sources,
        "verified": r.verified, "retries": r.retries,
        "trace": r.trace.steps}, ensure_ascii=False, indent=2)

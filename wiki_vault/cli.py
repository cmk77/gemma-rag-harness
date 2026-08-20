"""CLI — 강연 실습 흐름과 1:1 대응하는 커맨드.

  python -m wiki_vault.cli onboard --name "Mozi" --org "ExampleCorp" \\
      --interests "RAG, LangGraph" [--main-vault /path/to/main]
  python -m wiki_vault.cli inbox                 # 인박스 분류 미리보기
  python -m wiki_vault.cli ingest [--abstract-only] [--commentary "..."]
  python -m wiki_vault.cli query "질문" [--save]
  python -m wiki_vault.cli lint --new 10_concepts/카드.md
  python -m wiki_vault.cli audit
  python -m wiki_vault.cli verify 10_concepts/카드.md
  python -m wiki_vault.cli reindex
"""

from __future__ import annotations

import argparse
import json
import sys

from .bootstrap import build_app, reindex_all
from .config import WikiSettings
from .query import result_to_json

_CORECONTEXT_TMPL = """# Core Context
- name: {name}
- org: {org}
- interests: {interests}
- purpose: 이 위키 볼트는 '학습용 교재(재료)' 저장소다.
  체화가 끝난 지식은 메인 볼트(모선)로 옮긴다.
"""

_RULES_TMPL = """# Wiki Rules (세션 시작 시 최우선 적용)
1. 카드 title/description은 영어, body는 한국어.
2. 본문 전체를 열기 전 항상 프론트매터를 먼저 읽는다(점진적 노출).
3. 근거 없는 사실을 카드에 쓰지 않는다. 출처(source)를 항상 남긴다.
4. 인제스트 시 사용자 #목적 태그를 반드시 승계한다.
5. 새 지식 유입 시 관련 구지식 lint(최신화)를 수행한다.
6. 메인 볼트(모선)는 읽기 전용 — 절대 수정하지 않는다.
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="wiki_vault")
    sub = p.add_subparsers(dest="cmd", required=True)

    ob = sub.add_parser("onboard", help="볼트 스캐폴딩 + 시스템 파일 생성")
    ob.add_argument("--name", default="user")
    ob.add_argument("--org", default="")
    ob.add_argument("--interests", default="")
    ob.add_argument("--main-vault", default="")

    sub.add_parser("inbox", help="인박스 스캔 + 그룹핑 제안(미실행)")

    ig = sub.add_parser("ingest", help="인박스 전체 인제스트")
    ig.add_argument("--abstract-only", action="store_true",
                    help="논문 등: 초록 수준만 인제스트(토큰 절약)")
    ig.add_argument("--commentary", default="",
                    help="사용자 코멘터리/실험 맥락(권장)")

    q = sub.add_parser("query", help="위키 질의")
    q.add_argument("question")
    q.add_argument("--save", action="store_true",
                   help="결과를 쿼리 히스토리 카드로 저장")
    q.add_argument("--json", action="store_true")

    ln = sub.add_parser("lint", help="특정 카드 기준 구지식 최신화")
    ln.add_argument("--new", required=True, help="새 카드 상대경로")

    sub.add_parser("audit", help="볼트 전수 감사")

    vf = sub.add_parser("verify", help="카드 사실성 재점검")
    vf.add_argument("path")

    sub.add_parser("reindex", help="볼트 전수 재색인")

    args = p.parse_args(argv)
    settings = WikiSettings()
    if getattr(args, "main_vault", ""):
        from pathlib import Path
        settings.main_root = Path(args.main_vault)

    # onboard/inbox/audit은 ES 없이도 동작
    need_es = args.cmd in ("ingest", "query", "lint", "reindex", "verify")
    try:
        app = build_app(settings, with_es=need_es)
    except Exception as e:
        print(f"[오류] 초기화 실패: {e}", file=sys.stderr)
        return 1

    if args.cmd == "onboard":
        app.vault.write_system_files(
            _CORECONTEXT_TMPL.format(name=args.name, org=args.org,
                                     interests=args.interests),
            _RULES_TMPL)
        print(f"온보딩 완료: {app.settings.wiki_root}")
        print("폴더: 0_inbox / 10_concepts / 20_entities / 30_notes / "
              "90_raw / _system")
        return 0

    if args.cmd == "inbox":
        items = app.ingest.scan_inbox()
        if not items:
            print("인박스가 비어 있습니다. 0_inbox에 md/txt/html을 넣으세요.")
            return 0
        groups = app.ingest.group_items(items)
        for g in groups:
            names = [items[i].title for i in g["item_ids"]]
            print(f"* {g['topic']}  <- {names}  ({g.get('reason', '')})")
        return 0

    if args.cmd == "ingest":
        results = []
        items = app.ingest.scan_inbox()
        by_id = {it.id: it for it in items}
        for g in app.ingest.group_items(items):
            members = [by_id[i] for i in g["item_ids"] if i in by_id]
            if members:
                results.append(app.ingest.ingest_group(
                    members, abstract_only=args.abstract_only,
                    extra_context=args.commentary))
        made = [c for r in results for c in r.cards]
        for c in made:
            print(f"+ {app.vault.rel(c.path)}  [{c.frontmatter.type}] "
                  f"{c.frontmatter.title}")
        print(f"카드 {len(made)}개 생성, 로우 소스 90_raw 이동 완료")
        return 0

    if args.cmd == "query":
        r = app.query.run(args.question)
        if args.json:
            print(result_to_json(r))
        else:
            print(r.answer)
            print("\n--- 출처 ---")
            for s in r.sources:
                print(f"- {s}")
            print(f"(verified={r.verified}, retries={r.retries})")
        if args.save:
            print(f"히스토리 저장: {app.query.save_as_note(r)}")
        return 0

    if args.cmd == "lint":
        card = app.vault.load_card(args.new)
        rep = app.lint.on_new_cards([card])
        print(rep.to_json())
        return 0

    if args.cmd == "audit":
        print(app.lint.audit().to_json())
        return 0

    if args.cmd == "verify":
        print(json.dumps(app.lint.verify(args.path),
                         ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "reindex":
        print(f"재색인 완료: {reindex_all(app)}건")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

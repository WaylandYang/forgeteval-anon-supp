"""Interactive author manual review of judge-rejected hand-crafted
v0.4 cases.  Used to compute manual-judge agreement and to decide
which rejections are genuine bench bugs vs. judge semantic-
abstraction limitations.

Output:
    data/manual_review_v04.json   per-case author verdict + tag

Tags:
    bench_bug      — case is genuinely ill-formed (fix in v0.4.1)
    judge_limit    — case is well-formed; judge over-rejected due to
                     literal-vs-semantic interpretation gap
    unclear        — ambiguous, mark for second-pass review

Usage:
    py scripts/manual_review_rejected.py
        --> walks through each judge-rejected case, prints it,
            asks for: verdict (wf/ill) + tag + one-line note.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

# UTF-8 stdout for multilingual case content (Hebrew, CJK, Cyrillic).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

LETHE_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LETHE_REPO))

JUDGE_OUT = Path(__file__).resolve().parent.parent / "data" / \
    "judge_precision_v04.json"
MANUAL_OUT = Path(__file__).resolve().parent.parent / "data" / \
    "manual_review_v04.json"


def load_cases_by_id():
    from bench.forgeteval.adversarial import ATTACK_CATEGORIES
    out = {}
    for cat, cases in ATTACK_CATEGORIES.items():
        for c in cases:
            out[c.id] = (cat, c)
    return out


def render_case(case_id, cat, c, judge_reason):
    print("=" * 78)
    print(f"  {case_id}    category: {cat}")
    print("=" * 78)
    print("setup_facts:")
    for f in c.setup_facts:
        print(f"  - {f}")
    print("mutations:")
    for m in c.mutations:
        print(f"  - {tuple(m)}")
    print(f"final_query:     {c.final_query!r}")
    print(f"must_contain:    {list(c.must_contain)}")
    print(f"must_not_contain:{list(c.must_not_contain)}")
    print()
    print(f"JUDGE SAID: ill-formed.")
    print(f"  reason: {judge_reason}")
    print()


def main():
    if not JUDGE_OUT.exists():
        print(f"ERROR: {JUDGE_OUT} not found.")
        print("Run scripts/judge_precision_v04.py first.")
        return
    data = json.loads(JUDGE_OUT.read_text(encoding="utf-8"))
    rejected = [r for r in data["per_case"] if r["well_formed"] is False]
    if not rejected:
        print("No judge-rejected cases to review.")
        return

    cases_by_id = load_cases_by_id()

    # Load any existing manual verdicts so we can resume.
    manual: dict = {}
    if MANUAL_OUT.exists():
        try:
            manual = json.loads(MANUAL_OUT.read_text(encoding="utf-8"))
        except Exception:
            manual = {}

    print(f"\nJudge rejected {len(rejected)} of {data['n_cases']} hand-crafted")
    print(f"v0.4 cases.  Reviewing each one (resume-supported; "
          f"{len(manual)} already labeled).\n")

    for r in rejected:
        cid = r["case_id"]
        if cid in manual:
            print(f"  [SKIP already labeled] {cid}: "
                  f"{manual[cid]['verdict']} ({manual[cid]['tag']})")
            continue
        if cid not in cases_by_id:
            print(f"  ! {cid} not found in bench, skipping")
            continue
        cat, c = cases_by_id[cid]
        render_case(cid, cat, c, r["reason"])
        print("Your call:")
        print("  v) verdict — 'wf' (well-formed) or 'ill' (ill-formed)")
        verdict = input("  > ").strip().lower()
        if verdict not in {"wf", "ill"}:
            print("  Skipping (input not wf/ill).")
            continue
        print("  t) tag — 'bench_bug' / 'judge_limit' / 'unclear'")
        tag = input("  > ").strip().lower()
        if tag not in {"bench_bug", "judge_limit", "unclear"}:
            tag = "unclear"
        print("  n) one-line note:")
        note = input("  > ").strip()
        manual[cid] = {
            "case_id": cid, "category": cat,
            "judge_verdict": False, "judge_reason": r["reason"],
            "author_verdict": verdict == "wf",
            "tag": tag, "note": note,
        }
        MANUAL_OUT.write_text(
            json.dumps(manual, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  saved.\n")

    # Summary.
    n = len(manual)
    n_agree = sum(1 for v in manual.values()
                  if v["author_verdict"] is False)  # both say ill
    n_disagree = n - n_agree
    by_tag: dict[str, int] = {}
    for v in manual.values():
        by_tag[v["tag"]] = by_tag.get(v["tag"], 0) + 1

    print("\n" + "=" * 78)
    print(f"Manual review summary: {n} rejected cases reviewed")
    print(f"  author agrees (ill-formed):   {n_agree}")
    print(f"  author disagrees (well-formed): {n_disagree}")
    print(f"  by tag: {by_tag}")
    print(f"\nWritten to {MANUAL_OUT}")


if __name__ == "__main__":
    main()

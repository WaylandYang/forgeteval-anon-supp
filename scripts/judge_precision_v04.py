"""Judge-precision validation on hand-crafted v0.4 adversarial cases.

These cases were authored by hand over six months and are known to be
well-formed by construction.  Running the Qwen-2.5-72B admission
judge on them estimates the judge's false-rejection rate
(1 - precision), which is the methodological counterpart to the
self-eval circularity defense in the paper.

Output: a JSON report with per-case judge verdicts + summary, and a
plain-text summary suitable to drop into §3.3 of paper.tex.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

# Force UTF-8 stdout so Hebrew / CJK / Cyrillic characters in judge
# replies don't choke the default Windows GBK codepage.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

LETHE_REPO = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LETHE_REPO))
sys.path.insert(0, str(SCRIPTS_DIR))

# Reuse the judge infra from the generation script.
from generate_adversarial_cases import (
    make_judge_llm, JUDGE_PROMPT, extract_json_object,
)

OUT = Path(__file__).resolve().parent.parent / "data" / "judge_precision_v04.json"


def main():
    from bench.forgeteval.adversarial import ATTACK_CATEGORIES
    judge = make_judge_llm()

    all_cases = []
    for cat, cases in ATTACK_CATEGORIES.items():
        for c in cases:
            all_cases.append((cat, c))
    print(f"Running judge on {len(all_cases)} hand-crafted v0.4 cases "
          f"across {len(ATTACK_CATEGORIES)} categories.", flush=True)

    results = []
    t0 = time.perf_counter()
    for i, (cat, c) in enumerate(all_cases, 1):
        case_for_judge = {
            "setup_facts": list(c.setup_facts),
            "mutations": [list(m) for m in c.mutations],
            "final_query": c.final_query,
            "must_contain": list(c.must_contain),
            "must_not_contain": list(c.must_not_contain),
        }
        prompt = JUDGE_PROMPT.format(
            case_json=json.dumps(case_for_judge, ensure_ascii=False, indent=2)
        )
        try:
            raw = judge(prompt)
            verdict = extract_json_object(raw)
            well_formed = bool(verdict.get("well_formed", False))
            reason = str(verdict.get("reason", ""))
        except Exception as e:
            well_formed = None
            reason = f"parse error: {type(e).__name__}: {e}"
        results.append({
            "case_id": c.id, "category": cat,
            "well_formed": well_formed, "reason": reason,
        })
        # Incrementally save so we don't lose progress on crash.
        if i % 10 == 0 or i == len(all_cases):
            OUT.write_text(json.dumps({
                "n_cases": len(all_cases), "completed": i,
                "per_case": results,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            print(f"  [{i:3}/{len(all_cases)}] {c.id:<40} -> "
                  f"{well_formed} ({reason[:60]})", flush=True)
        except UnicodeEncodeError:
            ascii_reason = reason.encode("ascii", "replace").decode("ascii")
            print(f"  [{i:3}/{len(all_cases)}] {c.id:<40} -> "
                  f"{well_formed} ({ascii_reason[:60]})", flush=True)

    wall = time.perf_counter() - t0
    admitted = sum(1 for r in results if r["well_formed"] is True)
    rejected = sum(1 for r in results if r["well_formed"] is False)
    errored = sum(1 for r in results if r["well_formed"] is None)
    print(f"\nDONE in {wall:.1f}s ({wall/len(all_cases):.2f}s/case)")
    print(f"Admitted (judge says well-formed): {admitted}/{len(all_cases)} "
          f"= {admitted/len(all_cases)*100:.1f}%")
    print(f"Rejected (judge says ill-formed):  {rejected}")
    print(f"Errored (parse/api failure):       {errored}")

    OUT.write_text(json.dumps({
        "n_cases": len(all_cases),
        "admitted": admitted,
        "rejected": rejected,
        "errored": errored,
        "wall_seconds": wall,
        "per_case": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")

    # Show rejected cases so we can eyeball them for false-rejection.
    if rejected > 0:
        print("\n=== Rejected cases (judge says ill-formed) ===\n")
        for r in results:
            if r["well_formed"] is False:
                print(f"  {r['case_id']}  [{r['category']}]")
                print(f"    reason: {r['reason']}\n")


if __name__ == "__main__":
    main()

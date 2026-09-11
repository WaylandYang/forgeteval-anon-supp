"""Repair 32 supersede cases in external_raw/contributed that lack new_text.

Contributors placed the new fact directly in setup_facts (per the brief's
"new fact can be in setup_facts" clause).  My lenient admission filter
auto-converted these to purge, which over-deletes when the target_query
is broad.

Fix heuristic: new_text = setup_facts entry containing the must_contain
substring (later position wins for cyclic / multi-state cases).  If no
must_contain or no setup_fact matches, fall back to the LAST setup_fact
(contributors usually placed the new fact at the end).

Output: external_raw/contributed_fixed (single standard JSON array)
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

SRC = Path(__file__).resolve().parent.parent / "external_raw" / "contributed"
DST = Path(__file__).resolve().parent.parent / "external_raw" / "contributed_fixed"


def parse_concat(path: Path) -> list[dict]:
    txt = path.read_text(encoding="utf-8")
    parts = re.split(r"\]\s*\[", txt)
    cases = []
    for i, p in enumerate(parts):
        if i == 0:
            p = p.rstrip().rstrip("]") + "]"
        elif i == len(parts) - 1:
            p = "[" + p.lstrip().lstrip("[")
        else:
            p = "[" + p + "]"
        cases.extend(json.loads(p))
    return cases


def find_new_text(case: dict) -> tuple[str, str]:
    """Return (new_text, reason) for an under-specified supersede."""
    setup = case["setup_facts"]
    mcs = case.get("must_contain", [])
    if mcs:
        # Find the LATEST setup_fact that contains any must_contain string.
        for fact in reversed(setup):
            for mc in mcs:
                if mc.lower() in fact.lower():
                    return fact, f"contains must_contain '{mc[:30]}'"
    # Fallback: last setup_fact.
    return setup[-1], "fallback: last setup_fact"


def main():
    cases = parse_concat(SRC)
    print(f"loaded {len(cases)} cases")
    fixed = 0
    for c in cases:
        mut = c["mutation"]
        if mut.get("op") != "supersede":
            continue
        if mut.get("new_text") or mut.get("target_text"):
            continue
        new_text, reason = find_new_text(c)
        mut["new_text"] = new_text
        mut["_auto_filled"] = reason
        fixed += 1
        if fixed <= 5:
            print(f"  {c['id']}:")
            print(f"    target_query: {mut.get('target_query','')[:80]}")
            print(f"    new_text [{reason}]: {new_text[:80]}")
    print(f"\nfixed {fixed} supersede cases")

    DST.write_text(json.dumps(cases, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()

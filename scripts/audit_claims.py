"""Cross-check every k/n claim in the paper against the run data.

check_paper_numbers.py verifies that a list of canonical percentages
appears somewhere. That does not catch the failure that actually
happened: an appendix stating "identifier_obfuscation 47 to 100%" beside
a table showing 61 to 5, because neither figure was on the canonical
list.

This walks the other direction. Every "k/n" in the paper is looked up
against every category count in every run file; a fraction that matches
nothing measured is reported with its surrounding line, for a human to
judge. Most hits are legitimate -- counts of annotators, of cases in a
sample, of anything -- so this is a worklist, not a verdict.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PAPER = ROOT / "paper" / "paper.tex"

# Denominators that identify a per-category or whole-suite claim.
SUITE_N = {36, 37, 38, 39, 40, 76, 77, 100, 110, 112, 132, 200, 242, 253,
           310, 345, 385, 447}


def measured_fractions():
    """Every (pass, total) the run files contain, with where it came from."""
    seen: dict[tuple[int, int], list[str]] = {}
    for f in sorted(DATA.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        stack = [(f.name, d)]
        while stack:
            origin, node = stack.pop()
            if not isinstance(node, dict):
                continue
            p = node.get("pass", node.get("overall_pass"))
            t = node.get("total", node.get("overall_total"))
            if isinstance(p, int) and isinstance(t, int) and t:
                seen.setdefault((p, t), []).append(origin)
            if "pass" in node and "fail" in node:
                p2 = node["pass"]
                n2 = p2 + node["fail"]
                na = node.get("na", 0)
                seen.setdefault((p2, n2), []).append(origin)
                seen.setdefault((p2, n2 + na), []).append(origin)
            for v in node.values():
                if isinstance(v, dict):
                    stack.append((origin, v))

        # Ecosystem summaries carry no suite-level total, so a claim about
        # a system's overall score matches nothing unless we aggregate.
        by = d.get("by_category")
        if isinstance(by, dict) and by:
            tp = te = ts = 0
            for v in by.values():
                if not isinstance(v, dict) or "pass" not in v:
                    continue
                p = v["pass"]
                if "total" in v:
                    fl, na = v["total"] - p, 0
                else:
                    fl, na = v.get("fail", 0), v.get("na", 0)
                tp += p
                te += p + fl
                ts += p + fl + na
            for t in (te, ts):
                if t:
                    seen.setdefault((tp, t), []).append(f.name + " [total]")
    return seen


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    seen = measured_fractions()
    text = PAPER.read_text(encoding="utf-8")
    lines = text.split("\n")

    unmatched = []
    checked = 0
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith("%"):
            continue
        for m in re.finditer(r"(?<![\d.])(\d{1,3})/(\d{1,3})(?![\d.])", line):
            p, t = int(m.group(1)), int(m.group(2))
            if t not in SUITE_N or p > t:
                continue
            checked += 1
            if (p, t) not in seen:
                unmatched.append((i, p, t, line.strip()[:96]))

    print("checked %d k/n claims against %d measured fractions"
          % (checked, len(seen)))
    if not unmatched:
        print("every one matches a measured value")
        return 0
    print("\n%d claims match nothing in data/ -- verify each:\n" % len(unmatched))
    for i, p, t, ctx in unmatched:
        print("  L%-5d %d/%-4d %s" % (i, p, t, ctx))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

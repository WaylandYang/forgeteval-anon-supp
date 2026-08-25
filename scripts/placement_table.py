"""Generate paper/tab_placement.tex from the runs.

This table was hand-written in paper.tex. Every other hand-written table
in this paper has gone stale at least once -- most recently the two
figures, which kept asserting a retracted result after the text had
withdrawn it -- so it is generated here instead, from the same run files
the rest of the numbers come from.

Rows are the ablation's cells. A row whose run file is missing is
skipped with a warning rather than carried over at its old value, which
is the failure mode this script exists to prevent.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from runs import resolve  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
P = "openrouter_hook_deepseek_deepseek-v4-flash_"

# (call site, what the model sees, run file)
ROWS = [
    ("nowhere", "---", P + "nollm_v07_probed.json"),
    ("write, annotating", "data",
     P + "inscribe_v07_probed.json"),
    (r"\quad + control plane reads it", "data",
     P + "inscribe-aware_v07_probed.json"),
    ("write, merging", "data, may edit",
     P + "merge-inscribe_v07_probed.json"),
    ("mutation", r"\textbf{the request}", P + "v07_probed.json"),
    ("both", "the request",
     P + "inscribe+mutation_v07_probed.json"),
]

CATS = ["identifier_obfuscation", "cross_lingual_identifier", "compound_fact"]
BOLD = {"mutation"}


def cell(d, cat):
    v = d["by_category"].get(cat)
    return "---" if not v else "%d/%d" % (v["pass"], v["total"])


def main():
    out = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{model called at} & \textbf{sees} & \textbf{overall}",
        r"  & \textbf{ident\_obf} & \textbf{cross\_ling} & \textbf{compound}\\",
        r"\midrule",
    ]
    missing = []
    for site, sees, fname in ROWS:
        f = resolve(fname)
        if not f.exists():
            missing.append(fname)
            continue
        d = json.loads(f.read_text(encoding="utf-8-sig"))
        rate = "%.1f\\,\\%%" % (100 * d["overall_pass"] / d["overall_total"])
        cells = [cell(d, c) for c in CATS]
        if site in BOLD:
            rate = r"\textbf{%s}" % rate
            cells = [r"\textbf{%s}" % c for c in cells]
        out.append("%-30s & %-22s & %s & %s & %s & %s\\\\"
                   % (site, sees, rate, *cells))
        st = d.get("placement_stats") or {}
        if st.get("inscribe_calls"):
            print("  %-30s annotated %d/%d, %d unparseable"
                  % (site, st["annotated"], st["inscribe_calls"],
                     st["failed"]), file=sys.stderr)
    out += [r"\bottomrule", r"\end{tabular}"]

    (ROOT / "paper" / "tab_placement.tex").write_text(
        "\n".join(out) + "\n", encoding="utf-8")
    print("wrote paper/tab_placement.tex (%d rows)" % (len(ROWS) - len(missing)))
    if missing:
        print("MISSING, row omitted:", file=sys.stderr)
        for m in missing:
            print("  " + m, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

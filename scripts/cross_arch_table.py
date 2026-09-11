"""Generate the cross-architecture comparison from the designated runs.

This table was hand-written, which is why it never moved when the runs
were re-measured under the raised token cap: its reference column was the
348/385 repeat that the paper's own footnote excludes from headline use,
its caption named DeepSeek-V3 where the runs are V4-Flash, and neither
column summed to the total printed beneath it (340 and 343 against 348
and 344). verify_all only diffs generated assets, so a hand-written table
is exactly the thing it cannot catch.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "scripts"))
from runs import resolve                                    # noqa: E402

CATS = ["substring_trap", "prefix_collision", "paraphrase_supersession",
        "negation_trap", "temporal_qualifier", "shared_attribute",
        "compound_fact", "identifier_obfuscation",
        "cross_lingual_identifier", "recursive_supersession"]

P = "openrouter_hook_deepseek_deepseek-v4-flash_"
COLS = [
    (r"\sysLethe{}$+$LLM", P + "v07_probed.json"),
    ("LangGraph$+$LLM", P + "langgraph_v07_probed.json"),
]
DET = [
    (r"\sysLethe{}", P + "nollm_v07_probed.json"),
    ("LangGraph", P + "langgraph_nollm_v07_probed.json"),
]


def load(fname):
    return json.loads(resolve(fname).read_text(encoding="utf-8-sig"))


def cell(node):
    p = node["pass"]
    n = node["total"] if "total" in node else p + node.get("fail", 0)
    return p, n


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    hooked = [load(f) for _, f in COLS]
    det = [load(f) for _, f in DET]

    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             r"\textbf{Category} & \textbf{%s} & \textbf{%s}\\"
             % (COLS[0][0], COLS[1][0]), r"\midrule"]
    tot = [0, 0]
    for c in CATS:
        cells = []
        for i, d in enumerate(hooked):
            p, n = cell(d["by_category"][c])
            tot[i] += p
            cells.append("%d/%d (%d\%%)" % (p, n, round(100 * p / n)))
        lines.append("%-26s & %s & %s \\\\"
                     % (c.replace("_", r"\_"), cells[0], cells[1]))
    lines.append(r"\midrule")

    # The totals are the column sums, not a number carried separately.
    for i, d in enumerate(hooked):
        assert tot[i] == d["overall_pass"], (
            "column %d sums to %d but the run reports %d"
            % (i, tot[i], d["overall_pass"]))
    lines.append(r"\textbf{Overall} & \textbf{%d/385 (%.1f\,\%%)} & "
                 r"\textbf{%d/385 (%.1f\,\%%)} \\"
                 % (tot[0], 100 * tot[0] / 385, tot[1], 100 * tot[1] / 385))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "tab_cross_arch.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("wrote paper/tab_cross_arch.tex")

    # The end-to-end mini-table above it, from the same four runs.
    rows = []
    for (lab, _), dd, hd in zip(DET, det, hooked):
        a = 100 * dd["overall_pass"] / 385
        b = 100 * hd["overall_pass"] / 385
        rows.append((lab, a, b, b - a))
        print("  %-12s %.1f -> %.1f  (%+.1f)" % (lab, a, b, b - a))
    print("  lift spread %.1f pt   absolute spread %.1f pt"
          % (abs(rows[0][3] - rows[1][3]), abs(rows[0][2] - rows[1][2])))

    ml = [r"\begin{tabular}{lccc}", r"\toprule",
          r"\textbf{Backend} & \textbf{deterministic} & \textbf{$+$hook} "
          r"& \textbf{$\Delta$}\\", r"\midrule"]
    labels = ["reference store", r"LangGraph \code{InMemoryStore}"]
    for lab, (_, a, b, d) in zip(labels, rows):
        ml.append("%s & %.1f & %.1f & $+%.1f$\\\\" % (lab, a, b, d))
    ml += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "tab_cross_arch_ends.tex").write_text(
        "\n".join(ml) + "\n", encoding="utf-8")
    print("wrote paper/tab_cross_arch_ends.tex")


if __name__ == "__main__":
    main()

"""Generate the Mem0 infer=False/infer=True comparison from the runs.

This comparison has been rewritten by hand three times tonight and been
wrong after two of them: first because the router was measured under a
1024-token cap of its own, then because the corrected figures were typed
in and superseded within the hour. Generating it ends that.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from runs import resolve  # noqa: E402

P = "openrouter_hook_deepseek_deepseek-v4-flash_"
CATS = ["substring_trap", "prefix_collision", "paraphrase_supersession",
        "negation_trap", "temporal_qualifier", "shared_attribute",
        "compound_fact", "identifier_obfuscation",
        "cross_lingual_identifier", "recursive_supersession"]


def main():
    a = json.loads(resolve(P + "mem0_nollm_v07_probed.json")
                   .read_text(encoding="utf-8-sig"))
    b = json.loads(resolve(P + "mem0-infer_v07_probed.json")
                   .read_text(encoding="utf-8-sig"))

    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"\textbf{Category} & \textbf{\code{infer=False}} & "
             r"\textbf{\code{infer=True}} & \textbf{$\Delta$}\\",
             r"\midrule"]
    for c in CATS:
        x, y = a["by_category"][c], b["by_category"][c]
        px = 100 * x["pass"] / x["total"]
        py = 100 * y["pass"] / y["total"]
        d = py - px
        lines.append("%-26s & %d/%d (%.0f) & %d/%d (%.0f) & $%s$%.0f\\\\"
                     % (c.replace("_", r"\_"), x["pass"], x["total"], px,
                        y["pass"], y["total"], py,
                        "+" if d >= 0 else "-", abs(d)))
    ra = 100 * a["overall_pass"] / a["overall_total"]
    rb = 100 * b["overall_pass"] / b["overall_total"]
    lines += [r"\midrule",
              r"\textbf{Overall} & \textbf{%d/%d (%.1f)} & \textbf{%d/%d (%.1f)} & $%s$%.1f\\"
              % (a["overall_pass"], a["overall_total"], ra,
                 b["overall_pass"], b["overall_total"], rb,
                 "+" if rb >= ra else "-", abs(rb - ra)),
              r"\bottomrule", r"\end{tabular}"]

    (ROOT / "paper" / "tab_mem0_delta.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("wrote paper/tab_mem0_delta.tex  (%.1f -> %.1f)" % (ra, rb))


if __name__ == "__main__":
    main()

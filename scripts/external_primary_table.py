"""Generate the external-subset table for the four primary systems.

This table was hand-written and mixed two scorings. Its \\sysLethe{}
columns came from the re-measured v07 runs while its LangGraph and
\\sysMem{} columns were still carried from data/external_subset_results
.json, which predates the survivor and probing requirements -- in that
file Letta scores 8/8 on external cross_lingual where the re-measured run
scores 0/8. All six columns are now measured under the same requirements
as everything else, which moved two of them: \\sysMem{} 22 to 20, and
LangGraph+LLM 39 to 38 with cross_lingual 5/8 to 3/8.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

P = "openrouter_hook_deepseek_deepseek-v4-flash_"

# short label -> result file, in table order
COLS = [
    (r"\sysLethe{}", P + "nollm_external_probed.json"),
    ("LangGraph", P + "langgraph_nollm_external_probed.json"),
    (r"\sysPalace{}", P + "mempalace_nollm_external_probed.json"),
    (r"\sysMem{}", P + "mem0_nollm_external_probed.json"),
    (r"\sysLethe{}{+}LLM", "external_v07_lethe_llm.json"),
    ("LangGraph{+}LLM", P + "langgraph_external_probed.json"),
]

# (display, key) -- prefix_collision is 5 cases, the rest 8
CATS = [
    ("substr\\_trap", "substring_trap"),
    ("prefix\\_coll", "prefix_collision"),
    ("paraphr\\_super", "paraphrase_supersession"),
    ("neg\\_trap", "negation_trap"),
    ("temp\\_qual", "temporal_qualifier"),
    ("shared\\_attr", "shared_attribute"),
    ("compound", "compound_fact"),
    ("ident\\_obf", "identifier_obfuscation"),
    ("cross\\_ling", "cross_lingual_identifier"),
    ("recursive", "recursive_supersession"),
]

BOLD = {"negation_trap", "compound_fact", "identifier_obfuscation"}


def check_percase_agrees(runs):
    """The released per-case file must print what this table prints.

    data/external_subset_results.json is the only per-case record of this
    subset and no table reads it, so it sat two systems behind the runs for
    a while -- MEM0 at 22 against 20 and LangGraph+LLM at 39 against 38.
    A reader recomputing from the release would have got numbers the paper
    does not print. scripts/sync_external_results.py rebuilds it; this
    fails the build if the two drift apart again.
    """
    f = DATA / "external_subset_results.json"
    if not f.exists():
        return
    systems = json.loads(f.read_text(encoding="utf-8-sig"))["systems"]
    NAME = {r"\sysLethe{}": "Lethe", "LangGraph": "LangGraph",
            r"\sysPalace{}": "MemPalace", r"\sysMem{}": "Mem0",
            r"\sysLethe{}{+}LLM": "Lethe+LLM",
            "LangGraph{+}LLM": "LangGraph+LLM"}
    bad = []
    for label, d in runs:
        key = NAME.get(label)
        if key is None or key not in systems:
            continue
        if systems[key]["n_pass"] != d.get("overall_pass"):
            bad.append("%s: file %d, run %d"
                       % (key, systems[key]["n_pass"], d.get("overall_pass")))
    if bad:
        raise SystemExit(
            "external_subset_results.json disagrees with the runs this table "
            "reads: %s.  Run scripts/sync_external_results.py."
            % "; ".join(bad))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    # MemPalace exposes no deletion primitive at all, so every case is
    # N/A rather than a failure -- the same rule as the in-house tables.
    na_all = {r"\sysPalace{}"}
    runs = []
    for label, fname in COLS:
        f = DATA / fname
        if not f.exists():
            raise SystemExit("missing %s" % fname)
        runs.append((label, json.loads(f.read_text(encoding="utf-8-sig"))))
    check_percase_agrees(runs)

    out = [r"\begin{tabular}{lcccccc}", r"\toprule",
           r"\textbf{Cat.} & " + " & ".join(
               r"\textbf{%s}" % lab for lab, _ in COLS) + r"\\",
           r"\midrule"]

    for disp, key in CATS:
        cells = []
        n = None
        for lab_i, d in runs:
            if lab_i in na_all:
                cells.append("N/A")
                continue
            v = d["by_category"].get(key)
            if v is None:
                cells.append("---")
                continue
            tot = v.get("total", v["pass"] + v.get("fail", 0))
            n = tot
            c = "%d/%d" % (v["pass"], tot)
            cells.append(r"\textbf{%s}" % c if key in BOLD else c)
        name = disp + (" (n=%d)" % n if n not in (None, 8) else "")
        out.append("%-22s & %s \\\\" % (name, " & ".join(cells)))

    out.append(r"\midrule")
    tot = [d["overall_pass"] for _, d in runs]
    n77 = [d["overall_total"] for _, d in runs]
    isna = [lab in na_all for lab, _ in COLS]
    out.append(r"\textbf{Overall} & " + " & ".join(
        ("0/0 [77 N/A]" if q else "%d/%d" % (p, n))
        for q, p, n in zip(isna, tot, n77)) + r" \\")
    out.append(r"\textbf{\%} & " + " & ".join(
        ("---" if q else "%.1f" % (100 * p / n))
        for q, p, n in zip(isna, tot, n77)) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]

    (ROOT / "paper" / "tab_external.tex").write_text(
        "\n".join(out) + "\n", encoding="utf-8")
    print("wrote paper/tab_external.tex")
    for (lab, _), p, n in zip(COLS, tot, n77):
        print("  %-20s %2d/%d = %.1f" % (lab, p, n, 100 * p / n))


if __name__ == "__main__":
    main()

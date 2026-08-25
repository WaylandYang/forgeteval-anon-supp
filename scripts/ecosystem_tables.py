"""Generate the per-system appendix tables from the ecosystem runs.

These were hand-written and did not move when the systems were
re-measured, so a reader comparing an appendix against the heatmap found
A-MEM at 38/38 on canonicalization in one place and 0/38 in the other.
Generating them removes the possibility.

The ecosystem runners report pass/fail/na rather than pass/total, and the
N/A count matters here: a category the system has no primitive for is not
a category it failed, and the two overall lines say so explicitly.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

CATS = ["substring_trap", "prefix_collision", "paraphrase_supersession",
        "negation_trap", "temporal_qualifier", "shared_attribute",
        "compound_fact", "identifier_obfuscation",
        "cross_lingual_identifier", "recursive_supersession"]

# display name -> (summary file, LaTeX label for the column)
SYSTEMS = {
    # A-MEM was re-measured through the main runner, which reports
    # pass/total; the others use the ecosystem runners and report
    # pass/fail/na. Both shapes are handled below.
    "amem": ("openrouter_hook_deepseek_deepseek-v4-flash_amem_v07_probed.json",
             r"\sysAmem{}"),
    "letta": ("adversarial_summary_letta.json", "Letta"),
    "letta_llm": ("adversarial_summary_letta_llm.json", "Letta$+$LLM"),
    "openmemory": ("adversarial_summary_openmemory.json", "OpenMemory"),
    "graphiti": ("adversarial_summary_graphiti.json", "Graphiti"),
}

REF = ("openrouter_hook_deepseek_deepseek-v4-flash_nollm_v07_probed.json",
       r"vs.\ \sysLethe{}")


def ref_pcts():
    d = json.loads((DATA / REF[0]).read_text(encoding="utf-8-sig"))
    by = d["by_category"]
    out = {c: 100 * by[c]["pass"] / by[c]["total"] for c in CATS}
    out["_overall"] = 100 * d["overall_pass"] / d["overall_total"]
    return out


COVERAGE = {"amem": "A-MEM", "letta": "Letta", "letta_llm": "Letta+LLM",
            "openmemory": "OpenMemory", "graphiti": "Graphiti"}


def na_by_category(key):
    """Cases this store cannot express, from the adapter's own API.

    Not from the run: the runners score an absent primitive as a
    failure, so every run reports na=0.
    """
    name = COVERAGE.get(key)
    if not name:
        return {}
    cov = json.loads((DATA / "coverage.json").read_text(encoding="utf-8"))
    return cov["na"].get(name, {}).get("by_category", {})


def emit(key):
    fname, label = SYSTEMS[key]
    f = DATA / fname
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8-sig"))
    by = d["by_category"]
    ref = ref_pcts()
    na_map = na_by_category(key)

    rows = []
    ev_p = ev_n = strict_p = strict_n = 0
    for c in CATS:
        v = by.get(c)
        if not v:
            rows.append((c, "--- [n/a]", ref[c]))
            continue
        p = v["pass"]
        if "total" in v:                    # main-runner shape
            fl = v["total"] - p
        else:                               # ecosystem-runner shape
            fl = v["fail"]
        na = na_map.get(c, v.get("na", 0))
        n = p + fl - na                     # evaluable denominator
        assert n >= p, "%s/%s: more passes than evaluable cases" % (key, c)
        ev_p += p
        ev_n += n
        strict_p += p
        strict_n += n + na
        cell = ("--- [%d N/A]" % na if n == 0
                else "%d/%d (%d\\%%)%s" % (p, n, round(100 * p / n),
                                           " [%d N/A]" % na if na else ""))
        if n and 100 * p / n >= 90:
            cell = r"\textbf{%s}" % cell
        rows.append((c, cell, ref[c]))

    out = [r"\begin{tabular}{lcc}", r"\toprule",
           r"\textbf{Category} & \textbf{%s} & \textbf{%s} \\" % (label, REF[1]),
           r"\midrule"]
    for c, cell, r in rows:
        out.append("%-26s & %s & %.0f\\,\\%% \\\\"
                   % (c.replace("_", r"\_"), cell, r))
    out += [r"\midrule",
            r"\textbf{Overall} (evaluable) & \textbf{%d/%d (%.1f\%%)} & %.1f\,\%% \\"
            % (ev_p, ev_n, 100 * ev_p / ev_n, ref["_overall"]),
            r"\textbf{Overall} (strict, N/A=fail) & \textbf{%d/%d (%.1f\%%)} & %.1f\,\%% \\"
            % (strict_p, strict_n, 100 * strict_p / strict_n, ref["_overall"]),
            r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def main():
    wrote = []
    for key in SYSTEMS:
        tex = emit(key)
        if tex is None:
            print("  no run for %s" % key, file=sys.stderr)
            continue
        p = ROOT / "paper" / ("tab_eco_%s.tex" % key)
        p.write_text(tex, encoding="utf-8")
        wrote.append(p.name)
    print("wrote " + ", ".join(wrote))


if __name__ == "__main__":
    main()

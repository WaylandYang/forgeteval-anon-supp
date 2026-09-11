"""Rewrite make_heatmap.py's DATA block from the run files.

Hand-maintained matrices in this repo have gone stale every time, so the
rows that have runs are written from them. The five systems still
awaiting a full re-measurement keep their published values and are
listed here explicitly, so that adding a run file is the only thing
needed to move one of them across.
"""
from __future__ import annotations

import json
import pathlib
import re

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from runs import resolve  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
P = "openrouter_hook_deepseek_deepseek-v4-flash_"

CATS = ["substring_trap", "prefix_collision", "paraphrase_supersession",
        "negation_trap", "temporal_qualifier", "shared_attribute",
        "compound_fact", "identifier_obfuscation",
        "cross_lingual_identifier", "recursive_supersession"]

ORDER = ["MemPalace", "Lethe", "Mem0", "LangGraph", "Letta", "OpenMemory",
         "Mem0+v3", "A-MEM", "Graphiti", "HippoRAG", "Letta+LLM",
         "Lethe+LLM", "LangGraph+LLM"]

RUNS = {
    "MemPalace": "mempalace_nollm_v07_probed.json",
    "Lethe": "nollm_v07_probed.json",
    "Mem0": "mem0_nollm_v07_probed.json",
    "LangGraph": "langgraph_nollm_v07_probed.json",
    "Mem0+v3": "mem0-infer_v07_probed.json",
    "A-MEM": "amem_v07_probed.json",
    "Lethe+LLM": "v07_probed.json",
    "LangGraph+LLM": "langgraph_v07_probed.json",
}

def hipporag_strict():
    r"""HippoRAG's row from its per-case file, on the same denominator as
    every other row.

    The row used to be typed. Three of its ten cells disagreed with the
    file underneath: substring_trap read 40 against a strict 22,
    negation_trap 3 against 2, and shared_attribute was marked N/A --
    though all forty of its cases raised, exactly like prefix_collision,
    which the same row scored 0. A case that raises has not forgotten
    anything; N/A is for a primitive the API cannot express under any
    composition (\S4), which is a different thing. Strict throughout.
    """
    d = json.loads((DATA / "hipporag_results_inhouse.json")
                   .read_text(encoding="utf-8-sig"))
    tally = {c: [0, 0] for c in CATS}
    for r in d["results"]:
        c = r["category"]
        if c in tally:
            tally[c][1] += 1
            tally[c][0] += bool(r["passed"])
    return [round(100 * p / n) if n else None for p, n in
            (tally[c] for c in CATS)]


# Ecosystem runners report pass/fail/na rather than pass/total.
ECO = {"OpenMemory": "adversarial_summary_openmemory.json",
       "Letta": "adversarial_summary_letta.json",
       "Letta+LLM": "adversarial_summary_letta_llm.json",
       "Graphiti": "adversarial_summary_graphiti.json"}


def eco_pcts(fname):
    d = json.loads((DATA / fname).read_text(encoding="utf-8-sig"))
    by = d["by_category"]
    out = []
    for c in CATS:
        v = by.get(c)
        if not v:
            out.append(None)
            continue
        n = v["pass"] + v["fail"] + v.get("na", 0)
        out.append(round(100 * v["pass"] / n) if n else 0)
    return out


def pcts(fname):
    # resolve() prefers the re-measured run. Reading the plain name here
    # plotted the capped Mem0 router against an appendix reporting the
    # uncapped one.
    d = json.loads(resolve(fname).read_text(encoding="utf-8-sig"))
    by = d["by_category"]
    out = []
    for c in CATS:
        v = by.get(c)
        out.append(round(100 * v["pass"] / v["total"])
                   if v and v["total"] else 0)
    return out


def main():
    lines = ["DATA = {"]
    for name in ORDER:
        if name in RUNS:
            vals, note = pcts(P + RUNS[name]), ""
        elif name in ECO and (DATA / ECO[name]).exists():
            vals, note = eco_pcts(ECO[name]), ""
        else:
            vals, note = hipporag_strict(), "   # strict, not re-run"
        cells = ", ".join("NA" if v is None else str(v) for v in vals)
        lines.append('    "%s":%s[%s],%s'
                     % (name, " " * (18 - len(name)), cells, note))
    lines.append("}")

    p = ROOT / "scripts" / "make_heatmap.py"
    s = p.read_text(encoding="utf-8")
    start = s.index("DATA = {")
    end = s.index("\n}\n", start) + 3
    p.write_text(s[:start] + "\n".join(lines) + "\n" + s[end:],
                 encoding="utf-8")
    measured = [n for n in ORDER if n in RUNS]
    print("wrote %d measured rows: %s" % (len(measured), ", ".join(measured)))
    print("HippoRAG: strict from its per-case file, not re-measured")


if __name__ == "__main__":
    main()

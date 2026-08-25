"""Generate the external-subset ecosystem table from the runs.

Its columns previously showed Letta, OpenMemory, Mem0+v3 and A-MEM at 8/8
on both canonicalization categories, measured before the survivor and
probing requirements existed, beside a main-suite table showing the same
systems at 0/38. Systems with a re-run on this subset are now read from
it; the rest are omitted rather than carried at a figure the requirements
would not produce.
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

SHORT = {"substring_trap": "substr", "prefix_collision": "prefix",
         "paraphrase_supersession": "paraphr", "negation_trap": "negation",
         "temporal_qualifier": "temporal", "shared_attribute": "shared",
         "compound_fact": "compound", "identifier_obfuscation": "ident\\_obf",
         "cross_lingual_identifier": "cross\\_ling",
         "recursive_supersession": "recursive"}

COLUMNS = [
    ("Letta", "adversarial_summary_letta_external.json"),
    ("OpenMemory", "adversarial_summary_openmemory_external.json"),
    ("Letta$+$LLM", "adversarial_summary_letta_llm_external.json"),
]


def cells(fname):
    f = DATA / fname
    if not f.exists():
        return None
    by = json.loads(f.read_text(encoding="utf-8-sig"))["by_category"]
    out = {}
    for c in CATS:
        v = by.get(c)
        if not v:
            out[c] = None
            continue
        n = v["pass"] + v.get("fail", 0) + v.get("na", 0)
        out[c] = (v["pass"], n)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cols = [(lab, cells(f)) for lab, f in COLUMNS]
    cols = [(lab, c) for lab, c in cols if c]
    if not cols:
        print("no external ecosystem runs found", file=sys.stderr)
        return 1

    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             r"\textbf{Category} & "
             + " & ".join(r"\textbf{%s}" % l for l, _ in cols) + r"\\",
             r"\midrule"]
    for c in CATS:
        row = []
        for _, d in cols:
            v = d.get(c)
            row.append("---" if not v else "%d/%d" % v)
        name = SHORT[c]
        if c in ("identifier_obfuscation", "cross_lingual_identifier"):
            name = r"\textbf{%s}" % name
        lines.append("%-22s & " % name + " & ".join(row) + r"\\")
    lines.append(r"\midrule")
    tot = []
    for _, d in cols:
        p = sum(v[0] for v in d.values() if v)
        n = sum(v[1] for v in d.values() if v)
        tot.append(r"\textbf{%d/%d}" % (p, n))
    lines.append(r"\textbf{Overall} & " + " & ".join(tot) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    (ROOT / "paper" / "tab_external_eco.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("wrote paper/tab_external_eco.tex (%d systems)" % len(cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Bring data/external_subset_results.json onto the runs the tables read.

The file is the only per-case record of the 77-case external subset, and it
was written before the survivor and probing requirements existed. Two of its
systems have been re-measured since and the file never caught up: MEM0 22
against a re-measured 20, and LangGraph+LLM 39 against 38 with
cross_lingual_identifier 5/8 against 3/8. Nothing in the paper reads this
file, so the tables were right and the released per-case data was not --
which is the worse way round, because a reader recomputing from the release
gets a number the paper does not print.

Every system with an authoritative run is rebuilt from that run's per-case
checkpoint. The rest keep what they had and are named in `provenance` so it
is visible which is which.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "external_subset_results.json"
P = "openrouter_hook_deepseek_deepseek-v4-flash_"

# system -> the run that is authoritative for it, same files the tables read
AUTHORITATIVE = {
    "Lethe":          P + "nollm_external_probed.json",
    "LangGraph":      P + "langgraph_nollm_external_probed.json",
    "MemPalace":      P + "mempalace_nollm_external_probed.json",
    "Mem0":           P + "mem0_nollm_external_probed.json",
    "LangGraph+LLM":  P + "langgraph_external_probed.json",
}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    doc = json.loads(OUT.read_text(encoding="utf-8-sig"))
    systems = doc["systems"]
    provenance, moved = {}, []

    for name, fname in AUTHORITATIVE.items():
        run = DATA / fname
        ck = run.with_name(run.stem + "_ckpt.jsonl")
        if not run.exists() or not ck.exists():
            print("  no authoritative run for %s (%s)" % (name, fname))
            continue
        verdicts = {}
        for line in ck.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                verdicts[r["id"]] = bool(r["ok"])

        block = systems[name]
        before = block["n_pass"]
        # keep each case's category from the block being replaced: the
        # checkpoint carries verdicts, not the category mapping
        cat_of = {r["id"]: r["category"] for r in block["per_case"]}
        per_case, by_cat = [], {}
        for cid in sorted(verdicts, key=lambda c: [r["id"] for r in block["per_case"]].index(c)
                          if c in cat_of else 1 << 30):
            ok = verdicts[cid]
            cat = cat_of.get(cid, "unmapped")
            per_case.append({"id": cid, "category": cat, "passed": ok, "error": None})
            b = by_cat.setdefault(cat, {"pass": 0, "total": 0, "na": 0})
            b["total"] += 1
            b["pass"] += ok

        block["per_case"] = per_case
        block["by_category"] = by_cat
        block["n_pass"] = sum(verdicts.values())
        block["n_eval"] = len(verdicts)
        block["n_na"] = 0
        block["rate"] = round(100 * block["n_pass"] / len(verdicts), 1)
        provenance[name] = fname
        if block["n_pass"] != before:
            moved.append("%s %d -> %d" % (name, before, block["n_pass"]))

    for name in systems:
        provenance.setdefault(name, "as originally written; no re-measured run")

    doc["provenance"] = provenance
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print("  rebuilt %d systems from their runs" % len(AUTHORITATIVE))
    for m in moved:
        print("    moved: " + m)
    if not moved:
        print("    nothing moved; the file already matched the runs")


if __name__ == "__main__":
    main()

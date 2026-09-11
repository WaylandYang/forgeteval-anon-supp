"""Split per-case verdicts into hand-crafted core vs LLM-drafted subsets.

Directly addresses the circularity concern: if 253/385 cases are
LLM-drafted (DeepSeek-V3) and the LLM hook is also DeepSeek-V3, do we
re-observe the same per-category pattern on the 132 hand-crafted-only
subset that the human authors wrote?

Output: per-system pass rate on hand-crafted core (132 cases) vs.
LLM-drafted (253 cases), aggregate and per-category.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "lethe"))
DATA = Path(__file__).resolve().parent.parent / "data"

from bench.forgeteval.adversarial import ATTACK_CATEGORIES  # noqa: E402
from bench.forgeteval.adversarial_generated import (  # noqa: E402
    ADVERSARIAL_GENERATED,
)


def build_id_sets() -> tuple[set[str], set[str], dict[str, str]]:
    llm_ids = {c.id for cases in ADVERSARIAL_GENERATED.values() for c in cases}
    hc_ids: set[str] = set()
    cat_of: dict[str, str] = {}
    for cat, cases in ATTACK_CATEGORIES.items():
        for c in cases:
            cat_of[c.id] = cat
            if c.id not in llm_ids:
                hc_ids.add(c.id)
    return hc_ids, llm_ids, cat_of


def load_verdicts() -> dict[str, dict[str, bool | None]]:
    """Returns {system_label: {case_id: passed | None for NA}}.
    Merges v0.5 (365-case) + v0.5.1 (20-case ident-obf extras) for the
    six primary systems; uses single-file 385-case results for ecosystem
    systems (amem, graphiti, letta, letta+llm).
    """
    out: dict[str, dict[str, bool | None]] = {}

    # ---- v0.5 365-case files: lethe / mempalace / langmem
    with open(DATA / "adversarial_results_v05.json", encoding="utf-8") as fp:
        for sys_rec in json.load(fp):
            adapter = sys_rec["adapter"]
            label = {"lethe": "Lethe", "mempalace": "MemPalace",
                     "langmem": "LangGraph"}.get(adapter, adapter)
            d: dict[str, bool | None] = {}
            for c in sys_rec.get("per_case", []):
                d[c["id"]] = c.get("passed")
            out[label] = d

    # ---- v0.5 mem0
    with open(DATA / "adversarial_results_v05_mem0.json", encoding="utf-8") as fp:
        rec = json.load(fp)
        recs = rec if isinstance(rec, list) else [rec]
        for sys_rec in recs:
            d = {c["id"]: c.get("passed") for c in sys_rec.get("per_case", [])}
            out["Mem0"] = d

    # ---- v0.5 langgraph+LLM
    with open(DATA / "adversarial_results_v05_langgraph_llm.json",
              encoding="utf-8") as fp:
        rec = json.load(fp)
        recs = rec if isinstance(rec, list) else [rec]
        for sys_rec in recs:
            d = {c["id"]: c.get("passed") for c in sys_rec.get("per_case", [])}
            out["LangGraph+LLM"] = d

    # ---- v0.5 lethe+LLM
    with open(DATA / "adversarial_results_with_llm_siliconflow.json",
              encoding="utf-8") as fp:
        rec = json.load(fp)
        recs = rec if isinstance(rec, list) else [rec]
        for sys_rec in recs:
            d = {c["id"]: c.get("passed") for c in sys_rec.get("per_case", [])}
            out["Lethe+LLM"] = d

    # ---- v0.5.1 identifier_obfuscation extras (20 cases × 6 systems)
    with open(DATA / "identifier_obfuscation_v051_results.json",
              encoding="utf-8") as fp:
        extras = json.load(fp)
    for sys_name, rec in extras["by_system"].items():
        target = out.get(sys_name)
        if target is None:
            continue
        for c in rec.get("results", []):
            target[c["case_id"]] = c["pass"]

    # ---- ecosystem 385-case full runs
    for fname, label in [
        ("adversarial_results_amem.json", "A-MEM"),
        ("adversarial_results_graphiti.json", "Graphiti"),
        ("adversarial_results_letta.json", "Letta"),
        ("adversarial_results_letta_llm.json", "Letta+LLM"),
    ]:
        with open(DATA / fname, encoding="utf-8") as fp:
            rec = json.load(fp)
        results = rec.get("results", [])
        out[label] = {c["case_id"]: c.get("passed") for c in results}

    return out


def rate(verdicts: dict[str, bool | None], id_set: set[str]) -> tuple[int, int, int]:
    """Return (pass, evaluable, n_na) over case-ids ∩ id_set.
    Cases missing from verdicts are skipped (not counted)."""
    p = e = na = 0
    for cid in id_set:
        if cid not in verdicts:
            continue
        v = verdicts[cid]
        if v is True:
            p += 1
            e += 1
        elif v is False:
            e += 1
        else:
            na += 1
    return p, e, na


def main():
    hc_ids, llm_ids, cat_of = build_id_sets()
    verdicts = load_verdicts()

    SYS_ORDER = ["MemPalace", "Lethe", "Mem0", "LangGraph", "Letta", "A-MEM",
                 "Graphiti", "Letta+LLM", "Lethe+LLM", "LangGraph+LLM"]

    print(f"hand-crafted core: {len(hc_ids)} cases   LLM-drafted: {len(llm_ids)} cases")
    print()
    print(f"{'system':18s} | {'HC pass':>14s} | {'LLM pass':>14s} | {'Δ (HC−LLM)':>11s}")
    print("-" * 70)
    table_rows = []
    for sys_name in SYS_ORDER:
        if sys_name not in verdicts:
            print(f"{sys_name:18s}  (missing)")
            continue
        v = verdicts[sys_name]
        ph, eh, nh = rate(v, hc_ids)
        pl, el, nl = rate(v, llm_ids)
        rh = ph / eh * 100 if eh else 0
        rl = pl / el * 100 if el else 0
        delta = rh - rl
        hc_str = f"{ph:3d}/{eh:<3d} ({rh:5.1f}%)"
        ll_str = f"{pl:3d}/{el:<3d} ({rl:5.1f}%)"
        print(f"{sys_name:18s} | {hc_str:>14s} | {ll_str:>14s} | {delta:+6.1f}pt")
        table_rows.append({
            "system": sys_name,
            "hc_pass": ph, "hc_eval": eh, "hc_na": nh, "hc_rate": rh,
            "llm_pass": pl, "llm_eval": el, "llm_na": nl, "llm_rate": rl,
            "delta": delta,
        })

    print()
    print("=== Per-category pass rate on hand-crafted core ONLY ===")
    cats = sorted(set(cat_of.values()))
    header = f"{'category':30s} " + " ".join(f"{s:>8s}" for s in SYS_ORDER)
    print(header)
    print("-" * len(header))
    per_cat: dict[str, dict[str, tuple[int, int]]] = {}
    for cat in cats:
        ids_in_cat = {cid for cid in hc_ids if cat_of[cid] == cat}
        row = [f"{cat:30s} "]
        per_cat[cat] = {}
        for s in SYS_ORDER:
            v = verdicts.get(s, {})
            p = sum(1 for cid in ids_in_cat if v.get(cid) is True)
            e = sum(1 for cid in ids_in_cat
                    if v.get(cid) is True or v.get(cid) is False)
            per_cat[cat][s] = (p, e)
            if e:
                row.append(f"{p:>2d}/{e:<2d} ")
            else:
                row.append("  N/A  ")
        print("".join(row))

    out = {
        "hand_crafted_n": len(hc_ids),
        "llm_drafted_n": len(llm_ids),
        "rows": table_rows,
        "per_category_hc": {
            cat: {s: list(per_cat[cat][s]) for s in SYS_ORDER} for cat in cats
        },
    }
    out_path = DATA / "handcrafted_subset_breakdown.json"
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2)
    print()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

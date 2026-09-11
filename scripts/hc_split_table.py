"""Generate the hand-crafted vs LLM-drafted split from per-case verdicts.

The table was a snapshot and did not move when the systems were
re-measured, so its Lethe row summed to the previous deterministic total.
Rows are now computed from the checkpoints, which is also the only way
the split stays consistent with Table 2 when a run is repeated.

Systems whose per-case verdicts are not in the release keep no row rather
than an old one.  The ecosystem stores do have per-case verdicts, in the
runner's own result files rather than a checkpoint, so they carry rows
too: dropping them left the discussion citing rows that no longer
existed.  A-MEM is read from the re-measured checkpoint, not the
ecosystem file, which predates the token-cap fix.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))

P = "openrouter_hook_deepseek_deepseek-v4-flash_"

# display label -> checkpoint holding that configuration's verdicts
ROWS = [
    (r"\sysPalace{}", P + "mempalace_nollm_v07_probed_ckpt.jsonl"),
    (r"\sysLethe{}", P + "nollm_v07_probed_ckpt.jsonl"),
    (r"\sysMem{}", P + "mem0_nollm_v07_probed_ckpt.jsonl"),
    ("LangGraph", P + "langgraph_nollm_v07_probed_ckpt.jsonl"),
    (r"\sysLethe{}$+$LLM", P + "v07_probed_mt3000repr3_ckpt.jsonl"),
    ("LangGraph$+$LLM", P + "langgraph_v07_probed_mt3000_ckpt.jsonl"),
    (r"\sysMem{}$+$v3", P + "mem0-infer_v07_probed_mt3000_ckpt.jsonl"),
    (r"\sysAmem{}", P + "amem_v07_probed_ckpt.jsonl"),
]

# The ecosystem runners write one result file per system instead of a
# checkpoint stream. Same content, different shape.
# The placement ablation, resolved to the re-measured runs.
ABLATION = [
    ("no model", P + "nollm_v07_probed_ckpt.jsonl"),
    ("annotation", P + "inscribe_v07_probed_mt3000_ckpt.jsonl"),
    ("readable annotation", P + "inscribe-aware_v07_probed_mt3000_ckpt.jsonl"),
    ("merge authority", P + "merge-inscribe_v07_probed_mt3000_ckpt.jsonl"),
    ("mutation", P + "v07_probed_mt3000repr3_ckpt.jsonl"),
    ("both", P + "inscribe+mutation_v07_probed_mt3000_ckpt.jsonl"),
]

ECO_ROWS = [
    ("Letta", "adversarial_results_letta.json"),
    ("Letta$+$LLM", "adversarial_results_letta_llm.json"),
    ("OpenMemory", "adversarial_results_openmemory.json"),
    ("Graphiti", "adversarial_results_graphiti.json"),
]


def hc_ids():
    """Case ids authored by hand: everything not in the generated labels."""
    lab = json.loads(
        (ROOT / "bench/forgeteval/adversarial_generated_labels.json")
        .read_text(encoding="utf-8"))
    from scripts.repair_cross_lingual_queries import build_suite
    suite, _ = build_suite()
    return {c.id for c in suite if c.id not in lab}, {c.id for c in suite}


def verdicts(fname):
    f = DATA / fname
    if not f.exists():
        return None
    out = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r["ok"]
    return out


def eco_verdicts(fname):
    """Same verdicts from an ecosystem runner's result file.

    Cases whose primitive the store lacks are recorded as failures, not
    excluded, so the denominators match every other row.
    """
    f = DATA / fname
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8-sig"))
    return {r["case_id"]: bool(r["passed"]) for r in d["results"]}


def main():
    hc, allids = hc_ids()
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"\textbf{System} & \textbf{hand-crafted} & "
             r"\textbf{LLM-drafted} & \textbf{$\Delta$}\\", r"\midrule"]
    for label, ck in [(a, b) for a, b in ROWS] +             [(a, b) for a, b in ECO_ROWS]:
        v = (eco_verdicts(ck) if ck.endswith(".json") else verdicts(ck))
        if v is None:
            print("  no verdicts for %s" % label, file=sys.stderr)
            continue
        a = [k for k in v if k in hc]
        b = [k for k in v if k not in hc]
        pa, pb = sum(v[k] for k in a), sum(v[k] for k in b)
        ra = 100 * pa / len(a) if a else 0
        rb = 100 * pb / len(b) if b else 0
        lines.append("%-22s & %d/%d (%.1f) & %d/%d (%.1f) & $%s$%.1f \\\\"
                     % (label, pa, len(a), ra, pb, len(b), rb,
                        "+" if ra >= rb else "-", abs(ra - rb)))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "tab_hc_split.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("wrote paper/tab_hc_split.tex (%d hand-crafted of %d)"
          % (len(hc), len(allids)))

    # The controlled ablation restricted to the hand-crafted core.
    # Appendix reading (iii) quotes these, so they are released rather
    # than computed once in prose.
    arms = {}
    for label, ck in ABLATION:
        v = verdicts(ck)
        if v is None:
            print("  no verdicts for arm %s" % label, file=sys.stderr)
            continue
        a = [k for k in v if k in hc]
        arms[label] = {"pass": sum(v[k] for k in a), "total": len(a)}
    (DATA / "hc_ablation.json").write_text(
        json.dumps({"note": "controlled placement ablation, hand-crafted "
                            "cases only", "by_arm": arms}, indent=2),
        encoding="utf-8")
    print("wrote data/hc_ablation.json (%d arms)" % len(arms))


if __name__ == "__main__":
    main()

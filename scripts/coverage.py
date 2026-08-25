"""Which cases each adapter cannot express, and the two denominators.

N/A is a property of the adapter, not of the run: a store either can
compose the operation out of its API or it cannot, and no amount of
re-running changes that. So the coverage split is computable from the
suite and the adapters without spending a single LLM call, and the
evaluable score follows from a pass count already measured -- a case a
store cannot attempt was counted as a failure either way, so excluding it
moves the denominator and not the numerator.

    strict     = pass / 385                    (absent primitive = failure)
    evaluable  = pass / (385 - n/a)            (absent primitive excluded)

The paper reports strict as the headline, because a deployment that
cannot express a deletion has not forgotten anything, and shows evaluable
beside it so the reader can see what the coverage gap costs.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from runs import resolve  # noqa: E402

P = "openrouter_hook_deepseek_deepseek-v4-flash_"

# system -> ops it cannot express through any composition of its API.
# Taken from the adapters: an op listed here raises NotImplementedError.
MISSING = {
    "MemPalace": {"supersede", "release", "purge"},
    "Lethe": set(),
    "Mem0": set(),
    "LangGraph": set(),
    "Lethe+LLM": set(),
    "LangGraph+LLM": set(),
    "Mem0+v3": set(),
    "A-MEM": {"release"},
    "Letta": {"release"},
    "Letta+LLM": {"release"},
    "OpenMemory": {"release"},
    "Graphiti": {"release"},
    "HippoRAG": {"release", "purge"},
}

RUNS = {
    "MemPalace": P + "mempalace_nollm_v07_probed.json",
    "Lethe": P + "nollm_v07_probed.json",
    "Mem0": P + "mem0_nollm_v07_probed.json",
    "LangGraph": P + "langgraph_nollm_v07_probed.json",
    "Mem0+v3": P + "mem0-infer_v07_probed.json",
    "A-MEM": P + "amem_v07_probed.json",
    "Lethe+LLM": P + "v07_probed.json",
    "LangGraph+LLM": P + "langgraph_v07_probed.json",
}


def na_counts():
    """Cases each system cannot attempt, by category and in total."""
    from scripts.repair_cross_lingual_queries import build_suite
    from bench.forgeteval.adversarial import case_to_attack_category

    suite, _ = build_suite()
    out = {}
    for system, missing in MISSING.items():
        if not missing:
            out[system] = {"total": 0, "by_category": {}}
            continue
        by, n = {}, 0
        for c in suite:
            ops = {m[0] for m in c.mutations}
            if ops & missing:
                cat = case_to_attack_category(c.id)
                by[cat] = by.get(cat, 0) + 1
                n += 1
        out[system] = {"total": n, "by_category": by}
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    na = na_counts()

    print("%-16s %9s %9s %9s" % ("system", "strict", "n/a", "evaluable"))
    rows = {}
    for system, f in RUNS.items():
        d = json.loads(resolve(f).read_text(encoding="utf-8-sig"))
        p, t = d["overall_pass"], d["overall_total"]
        n = na[system]["total"]
        ev = t - n
        rows[system] = {"pass": p, "strict_total": t, "na": n,
                        "evaluable_total": ev,
                        "strict_pct": round(100 * p / t, 1),
                        "evaluable_pct": round(100 * p / ev, 1) if ev else None}
        ev_s = ("%d/%d (%.1f)" % (p, ev, 100 * p / ev)) if ev else "--- (0 cases)"
        print("%-16s %9s %9d %14s"
              % (system, "%d/%d" % (p, t), n, ev_s))

    (ROOT / "data" / "coverage.json").write_text(
        json.dumps({"na": na, "scores": rows}, indent=2), encoding="utf-8")
    print("\nwrote data/coverage.json")


if __name__ == "__main__":
    main()

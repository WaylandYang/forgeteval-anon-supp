"""Recompute every paired test the paper quotes, from the designated runs.

The released data/mcnemar_significance.json described a
"Lethe deterministic vs Lethe+LLM hook, DeepSeek-V3" comparison with
a_pass 244 and b_pass 353 -- pre-repair, pre-re-measurement, and a
different hook model. The paper cites it as the full test output, so a
reviewer opening it found numbers that appear nowhere in the paper.

Exact McNemar throughout (binomial on the discordant pairs), with
Holm-Bonferroni across the ten categories within each comparison.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from math import comb

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from runs import resolve  # noqa: E402

P = "openrouter_hook_deepseek_deepseek-v4-flash_"

COMPARISONS = [
    ("Lethe deterministic vs Lethe+LLM (mutation-time hook, V4-Flash)",
     "nollm_v07_probed", "v07_probed"),
    ("LangGraph deterministic vs LangGraph+LLM",
     "langgraph_nollm_v07_probed", "langgraph_v07_probed"),
    ("Lethe deterministic vs LangGraph deterministic",
     "nollm_v07_probed", "langgraph_nollm_v07_probed"),
    ("Lethe deterministic vs Mem0 infer=False",
     "nollm_v07_probed", "mem0_nollm_v07_probed"),
]


def verdicts(name):
    s = resolve(P + name + ".json")
    p = s.with_name(s.name.replace(".json", "_ckpt.jsonl"))
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = bool(r["ok"])
    return out


def category(cid):
    m = re.match(r"adv_(.+)_\d+$", cid)
    return m.group(1) if m else "other"


def exact_mcnemar(b, c):
    """Two-sided exact binomial on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def holm(pairs):
    """Holm-Bonferroni; pairs is [(key, p)]."""
    order = sorted(pairs, key=lambda kv: kv[1])
    m = len(order)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(order):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    results = []
    for label, fa, fb in COMPARISONS:
        A, B = verdicts(fa), verdicts(fb)
        ids = sorted(set(A) & set(B))
        b = sum(1 for i in ids if A[i] and not B[i])   # a-only
        c = sum(1 for i in ids if B[i] and not A[i])   # b-only
        entry = {
            "label": label,
            "n_paired": len(ids),
            "a_pass": sum(A[i] for i in ids),
            "b_pass": sum(B[i] for i in ids),
            "overall": {"a_only": b, "b_only": c, "discordant": b + c,
                        "p_exact": exact_mcnemar(b, c)},
            "by_category": {},
        }
        raw = {}
        for cat in sorted({category(i) for i in ids}):
            sub = [i for i in ids if category(i) == cat]
            bb = sum(1 for i in sub if A[i] and not B[i])
            cc = sum(1 for i in sub if B[i] and not A[i])
            p = exact_mcnemar(bb, cc)
            raw[cat] = p
            entry["by_category"][cat] = {
                "n": len(sub), "a_pass": sum(A[i] for i in sub),
                "b_pass": sum(B[i] for i in sub),
                "a_only": bb, "b_only": cc, "discordant": bb + cc,
                "p_exact": p,
            }
        for cat, ph in holm(list(raw.items())).items():
            entry["by_category"][cat]["p_holm"] = ph
        results.append(entry)

        print("%s" % label)
        print("    %d/%d vs %d/%d   discordant %d (%d a-only, %d b-only)"
              "   p = %.3g"
              % (entry["a_pass"], entry["n_paired"], entry["b_pass"],
                 entry["n_paired"], b + c, b, c, entry["overall"]["p_exact"]))
        surv = [k for k, v in entry["by_category"].items()
                if v["p_holm"] < 0.05]
        print("    categories surviving Holm: %s" % (surv or "none"))

    (ROOT / "data" / "mcnemar_significance.json").write_text(
        json.dumps({"scoring": "probe-based, survivor and one-form "
                               "requirements in force",
                    "test": "exact McNemar, Holm-Bonferroni within "
                            "each comparison",
                    "comparisons": results}, indent=1),
        encoding="utf-8")
    print("wrote data/mcnemar_significance.json")


if __name__ == "__main__":
    main()

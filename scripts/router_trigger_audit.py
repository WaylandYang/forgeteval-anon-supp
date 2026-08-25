"""How close does the escalation router get to the oracle it approximates?

Measured without spending a single model call: replay every case against a
deterministic store, record which mutations the router *would* escalate,
and compare that set against the cases where the hook actually changed the
verdict (data/openrouter_hook_*_v07_probed_ckpt.jsonl).

Recall is the number that matters. A missed escalation costs a case; a
false one costs a fraction of a cent, so the router is tuned to
over-trigger and the interesting quantity is what fraction of the oracle's
22.3% call budget it has to spend to reach a given recall.

  python scripts/router_trigger_audit.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"

from fastembed import TextEmbedding  # noqa: E402

from bench.forgeteval.adversarial import case_to_attack_category  # noqa: E402
from bench.forgeteval.router import (  # noqa: E402
    EscalationRouter, RoutedLetheAdapter,
)
from bench.forgeteval.scoring import run_scored  # noqa: E402
from scripts.repair_cross_lingual_queries import build_suite  # noqa: E402

DET = "openrouter_hook_deepseek_deepseek-v4-flash_nollm_v07_probed_ckpt.jsonl"
HOOK = "openrouter_hook_deepseek_deepseek-v4-flash_v07_probed_ckpt.jsonl"


def verdicts(name):
    return {json.loads(l)["id"]: bool(json.loads(l)["ok"])
            for l in (DATA / name).read_text(encoding="utf-8-sig").splitlines()
            if l.strip()}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    det, hook = verdicts(DET), verdicts(HOOK)
    need = {i for i in det if hook.get(i) and not det[i]}
    hurt = {i for i in det if det[i] and not hook.get(i, False)}

    emb = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    embedder = lambda t: list(next(iter(emb.embed([t]))))  # noqa: E731
    suite, _ = build_suite()

    fired, by_trigger = set(), Counter()
    for c in suite:
        r = EscalationRouter()
        a = RoutedLetheAdapter(embedder=embedder, vector_dim=384,
                               llm=None, router=r)
        try:
            run_scored(c, a, probed=True)
        except Exception:
            pass
        if r.stats["escalated"]:
            fired.add(c.id)
            for k in ("cross_script", "variant_family", "compound_row"):
                if r.stats[k]:
                    by_trigger[k] += 1

    n = len(suite)
    tp, fp, fn = len(fired & need), len(fired - need), len(need - fired)
    print(f"oracle escalation set        {len(need)}/{n} "
          f"({len(need)/n:.1%} of calls)")
    print(f"router escalation set        {len(fired)}/{n} "
          f"({len(fired)/n:.1%} of calls)")
    print(f"  hit {tp}   false {fp}   missed {fn}")
    print(f"  recall {tp/max(len(need),1):.1%}   "
          f"precision {tp/max(len(fired),1):.1%}")
    print(f"  cases the hook breaks, correctly skipped: "
          f"{len(hurt - fired)}/{len(hurt)}")

    print("\ncases firing each trigger (a case may fire more than one):")
    for k, v in by_trigger.most_common():
        print(f"  {k:<18}{v:>4}")

    print("\nmissed, by category:")
    for k, v in Counter(case_to_attack_category(i)
                        for i in (need - fired)).most_common():
        print(f"  {k:<28}{v:>3}")

    out = {"oracle_set": sorted(need), "router_set": sorted(fired),
           "recall": tp / max(len(need), 1),
           "precision": tp / max(len(fired), 1),
           "call_fraction": len(fired) / n,
           "oracle_call_fraction": len(need) / n,
           "by_trigger": dict(by_trigger)}
    (DATA / "router_trigger_audit.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print("\nwrote data/router_trigger_audit.json")


if __name__ == "__main__":
    main()

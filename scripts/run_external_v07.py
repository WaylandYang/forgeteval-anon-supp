"""Re-measure the external-authored subset on the repaired cases.

The 77 externally written cases carry the same must_contain = [] defect as
the shipped in-house suite (\\S validity), so the replication the paper
leans on -- "deterministic 0/8, every LLM-hook 8/8" on identifier
obfuscation -- was scored the same way nuke scores 8/8. This runs the
repaired cases (repair_external_subset.py) through the same adapters and
the same probe scorer used everywhere else, so the external evidence is
held to the standard the in-house evidence now is.

  OPENROUTER_API_KEY=... python scripts/run_external_v07.py --adapter lethe
  OPENROUTER_API_KEY=... python scripts/run_external_v07.py --adapter lethe+llm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"

from bench.forgeteval.adapter import LetheAdapter  # noqa: E402
from bench.forgeteval.scoring import run_scored  # noqa: E402

KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash")
BASE = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")


@dataclass
class Case:
    """The subset of GeneratedCase that run_scored touches."""
    id: str
    category: str
    setup_facts: list
    mutations: list
    final_query: str
    must_contain: list = field(default_factory=list)
    must_not_contain: list = field(default_factory=list)


def make_llm():
    import openai
    client = openai.OpenAI(api_key=KEY, base_url=BASE)
    usage = {"calls": 0, "errors": 0}

    def llm(prompt):
        try:
            r = client.chat.completions.create(
                model=MODEL, max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "3000")), temperature=0.0,
                messages=[{"role": "user", "content": prompt}])
            usage["calls"] += 1
            return r.choices[0].message.content or ""
        except Exception as e:
            usage["errors"] += 1
            print(f"  [llm err] {type(e).__name__}: {str(e)[:90]}")
            return ""
    return llm, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", choices=["lethe", "lethe+llm"],
                    default="lethe")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    src = json.loads((DATA / "external_subset_cases_v07.json")
                     .read_text(encoding="utf-8-sig"))
    cases = [Case(id=c["id"], category=c["category"],
                  setup_facts=c["setup_facts"],
                  mutations=[tuple(m) for m in c["mutations"]],
                  final_query=c["final_query"],
                  must_contain=c.get("must_contain", []),
                  must_not_contain=c.get("must_not_contain", []))
             for c in src["admitted_cases"]]

    from fastembed import TextEmbedding
    em = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    emb = lambda t: list(next(iter(em.embed([t]))))  # noqa: E731
    usage = {"calls": 0, "errors": 0}
    llm = None
    if args.adapter == "lethe+llm":
        if not KEY:
            sys.exit("set OPENROUTER_API_KEY")
        llm, usage = make_llm()
    adapter = LetheAdapter(embedder=emb, vector_dim=384, llm=llm)

    by_cat = defaultdict(lambda: {"pass": 0, "total": 0})
    npass, t0 = 0, time.perf_counter()
    for i, c in enumerate(cases):
        try:
            ok = run_scored(c, adapter, probed=True)
        except Exception as e:
            ok = False
            print(f"  [case err] {c.id}: {type(e).__name__}: {str(e)[:70]}")
        by_cat[c.category]["total"] += 1
        by_cat[c.category]["pass"] += ok
        npass += ok
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(cases)}  pass={npass}", flush=True)

    slug = args.adapter.replace("+", "_")
    print(f"\n=== external v0.7 ({args.adapter}) "
          f"{npass}/{len(cases)} = {npass/len(cases):.1%} "
          f"({time.perf_counter()-t0:.0f}s, {usage['calls']} calls, "
          f"{usage['errors']} err) ===")
    for cat in sorted(by_cat):
        d = by_cat[cat]
        print(f"  {cat:<28}{d['pass']:>3}/{d['total']:<3} "
              f"{d['pass']/d['total']:.0%}")

    (DATA / f"external_v07_{slug}.json").write_text(json.dumps(
        {"adapter": args.adapter, "model": MODEL if llm else None,
         "suite": "external-77-v0.7", "scorer": "probe",
         "overall_pass": npass, "overall_total": len(cases),
         "by_category": dict(by_cat), "usage": usage}, indent=1),
        encoding="utf-8")
    print(f"wrote data/external_v07_{slug}.json")


if __name__ == "__main__":
    main()

"""Measure the two- to three-surface-form step, with the gate first.

The external subset showed the hook at 0/8 on three-form cross-lingual
cases against 32/38 on our two-form ones, and the paper reads the second
as an upper bound for the first. That concedes a limit without locating
it: the external cases differ from ours in author as well as in form
count, so the drop could be either.

These pairs differ in one thing. Same eight entities, same eight script
families, same sibling that must survive, same request naming one form --
only the number of surfaces the target carries changes.

The gate runs first, as it does for every requirement in this paper: an
oracle deleting exactly the forbidden rows must still pass, and null and
nuke must not. A condition an oracle cannot pass is unsatisfiable rather
than hard, and would say nothing about the hook.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bench.forgeteval.generate import GeneratedCase        # noqa: E402
from bench.forgeteval.scoring import run_scored            # noqa: E402
from bench.forgeteval.adapter import LetheAdapter          # noqa: E402
from run_degenerate_baselines import NullAdapter, NukeAdapter  # noqa: E402
from external_gate import OracleAdapter                    # noqa: E402


def to_cases(raw):
    return [GeneratedCase(
        id=c["id"], family=c["category"], setup_facts=c["setup_facts"],
        mutations=[tuple(m) for m in c["mutations"]],
        final_query=c["final_query"], must_contain=c["must_contain"],
        must_not_contain=c["must_not_contain"]) for c in raw]


def score(adapter, cases, arm=False):
    n = 0
    for c in cases:
        adapter.reset()
        if arm:
            adapter.arm(c.must_not_contain)
        if run_scored(c, adapter, probed=True):
            n += 1
    return n


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    raw = json.loads((ROOT / "data" / "threeform_cases.json")
                     .read_text(encoding="utf-8"))
    conds = [("two-form", to_cases(raw["two_form"])),
             ("three-form", to_cases(raw["three_form"])),
             ("exonym", to_cases(raw["exonym"]))]

    print("=== gate ===")
    gate_ok = True
    for label, cases in conds:
        o = score(OracleAdapter(), cases, arm=True)
        nu = score(NullAdapter(), cases)
        nk = score(NukeAdapter(), cases)
        print("  %-11s oracle %d/%d   null %d   nuke %d"
              % (label, o, len(cases), nu, nk))
        if o != len(cases) or nu or nk:
            gate_ok = False
    if not gate_ok:
        print("\nGATE FAILED -- a condition an oracle cannot pass, or that a "
              "non-system can, measures nothing about the hook.")
        return 1

    key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("\nset LLM_API_KEY to measure the deterministic/hook pair")
        return 0

    from run_openrouter_hook import make_llm
    from fastembed import TextEmbedding
    em = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    emb = lambda t: list(next(iter(em.embed([t]))))     # noqa: E731
    llm, usage = make_llm()

    print("\n=== measured ===")
    out = {}
    for label, cases in conds:
        det = score(LetheAdapter(embedder=emb, vector_dim=384, llm=None), cases)
        hook = score(LetheAdapter(embedder=emb, vector_dim=384, llm=llm), cases)
        out[label] = {"n": len(cases), "deterministic": det, "hook": hook}
        print("  %-11s deterministic %d/%d   hook %d/%d"
              % (label, det, len(cases), hook, len(cases)))
    out["llm_errors"] = usage["errors"]
    (ROOT / "data" / "threeform_results.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote data/threeform_results.json  (llm errors %d)"
          % usage["errors"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

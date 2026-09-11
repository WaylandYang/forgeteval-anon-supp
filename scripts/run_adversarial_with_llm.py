"""Run ForgetEval-Adv with a real Anthropic LLM wired into LetheAdapter.

Requires:
    export ANTHROPIC_API_KEY=sk-ant-...
    pip install anthropic

Default model: claude-haiku-4-5-20251001 (small + cheap; the prompts are
narrow and the contract is JSON-shaped, so haiku is sufficient).

The LLM is invoked at most once per `supersede` and once per `purge`
operation.  Lethe's recall hot path is never asked to consult the model
— that's the architectural invariant.

Output: ../data/adversarial_results_with_llm.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

LETHE_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LETHE_REPO))

OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)

MODEL = os.environ.get("LETHE_LLM_MODEL", "claude-haiku-4-5-20251001")


def make_anthropic_llm():
    """Return a Callable[[str], str] that proxies prompt→response through
    Anthropic's Messages API.  Caches results in-memory to avoid paying
    for duplicate calls if the runner is restarted mid-suite."""
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit(
            "ANTHROPIC_API_KEY not set.  Export it before running, e.g.:\n"
            "    export ANTHROPIC_API_KEY=sk-ant-..."
        )
    client = anthropic.Anthropic(api_key=key)
    cache: dict[str, str] = {}

    def llm(prompt: str) -> str:
        if prompt in cache:
            return cache[prompt]
        resp = client.messages.create(
            model=MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in resp.content
                       if getattr(block, "type", None) == "text")
        cache[prompt] = text
        return text

    return llm


def run(adapter, name: str) -> dict:
    from bench.forgeteval.adversarial import (
        ADVERSARIAL_TESTS, ATTACK_CATEGORIES, case_to_attack_category,
    )

    results = []
    t0 = time.perf_counter()
    for i, case in enumerate(ADVERSARIAL_TESTS, 1):
        cat = case_to_attack_category(case.id)
        try:
            passed = case.run(adapter)
            err = None
        except NotImplementedError:
            passed = False
            err = "N/A (capability not supported)"
        except Exception as e:
            passed = False
            err = f"{type(e).__name__}: {e}"
        results.append({
            "id": case.id,
            "family": case.family,
            "attack_category": cat,
            "passed": passed,
            "error": err,
        })
        print(f"  [{i:>2}/64] {('pass' if passed else 'fail'):<4} {case.id}",
              flush=True)
    wall = time.perf_counter() - t0

    by_category = {}
    for cat in ATTACK_CATEGORIES:
        rows = [r for r in results if r["attack_category"] == cat]
        passed = sum(1 for r in rows if r["passed"])
        by_category[cat] = {"pass": passed, "total": len(rows),
                            "rate": passed / max(len(rows), 1)}

    by_family = defaultdict(lambda: {"pass": 0, "total": 0})
    for r in results:
        by_family[r["family"]]["total"] += 1
        if r["passed"]:
            by_family[r["family"]]["pass"] += 1
    for fam, d in by_family.items():
        d["rate"] = d["pass"] / max(d["total"], 1)

    total_pass = sum(r["passed"] for r in results)
    return {
        "adapter": name,
        "model": MODEL,
        "suite": "adversarial",
        "case_count": len(results),
        "overall_pass": total_pass,
        "overall_total": len(results),
        "overall_rate": total_pass / max(len(results), 1),
        "by_family": dict(by_family),
        "by_attack_category": by_category,
        "per_case": results,
        "wall_seconds": wall,
    }


def main():
    from bench.forgeteval.adapter import LetheAdapter
    from fastembed import TextEmbedding

    embed_model = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"loading embedder: {embed_model}", flush=True)
    model = TextEmbedding(embed_model)
    def embedder(text):
        return list(next(iter(model.embed([text]))))

    print(f"using LLM model: {MODEL}", flush=True)
    llm = make_anthropic_llm()
    adapter = LetheAdapter(embedder=embedder, vector_dim=384, llm=llm)

    print("running lethe + llm on adversarial suite (64 cases)...", flush=True)
    r = run(adapter, "lethe+llm")
    print(f"\noverall = {r['overall_rate']:.4f} "
          f"({r['overall_pass']}/{r['overall_total']})  "
          f"wall={r['wall_seconds']:.1f}s\n", flush=True)
    print("by attack category:")
    for cat, d in r["by_attack_category"].items():
        print(f"  {cat:<28} {d['pass']:>2}/{d['total']:<2} ({d['rate']:.0%})")

    out = OUT / "adversarial_results_with_llm.json"
    out.write_text(json.dumps([r], indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

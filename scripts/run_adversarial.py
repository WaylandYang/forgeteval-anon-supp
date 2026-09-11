"""Run the adversarial suite (64 cases, 8 attack categories) for the paper.

Captures per-case pass/fail and aggregates by (family, attack_category)
to data/adversarial_results.json.

Currently runs Lethe only.  Mem0 / MemPalace will need a follow-up run.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

LETHE_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LETHE_REPO))

OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)


def run(adapter, name: str) -> dict:
    from bench.forgeteval.adversarial import (
        ADVERSARIAL_TESTS, ATTACK_CATEGORIES, case_to_attack_category,
    )

    results = []
    t0 = time.perf_counter()
    for case in ADVERSARIAL_TESTS:
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
    wall = time.perf_counter() - t0

    by_category: dict[str, dict] = {}
    for cat in ATTACK_CATEGORIES:
        rows = [r for r in results if r["attack_category"] == cat]
        passed = sum(1 for r in rows if r["passed"])
        by_category[cat] = {"pass": passed, "total": len(rows),
                            "rate": passed / max(len(rows), 1)}

    by_family: dict[str, dict] = defaultdict(lambda: {"pass": 0, "total": 0})
    for r in results:
        by_family[r["family"]]["total"] += 1
        if r["passed"]:
            by_family[r["family"]]["pass"] += 1
    for fam, d in by_family.items():
        d["rate"] = d["pass"] / max(d["total"], 1)

    total_pass = sum(r["passed"] for r in results)
    return {
        "adapter": name,
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


def _print_summary(r: dict) -> None:
    print(f"  {r['adapter']:<10} overall = {r['overall_rate']:.4f} "
          f"({r['overall_pass']}/{r['overall_total']})  "
          f"wall={r['wall_seconds']:.1f}s", flush=True)
    for cat, d in r["by_attack_category"].items():
        print(f"    {cat:<28} {d['pass']:>2}/{d['total']:<2} ({d['rate']:.0%})",
              flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", default=["lethe"],
                    choices=["lethe", "mem0", "mempalace", "langmem"],
                    help="Which adapters to evaluate (one or many).")
    ap.add_argument("--output", default="adversarial_results.json",
                    help="Output JSON filename inside data/")
    ap.add_argument("--embedder",
                    default="sentence-transformers/all-MiniLM-L6-v2",
                    help="HuggingFace embedder name (default English MiniLM)")
    args = ap.parse_args()

    embed_model = args.embedder
    print(f"loading embedder: {embed_model}", flush=True)
    from fastembed import TextEmbedding
    model = TextEmbedding(embed_model)
    def embedder(text: str) -> list[float]:
        return list(next(iter(model.embed([text]))))

    runs = []

    if "lethe" in args.adapters:
        from bench.forgeteval.adapter import LetheAdapter
        print("\nrunning lethe...", flush=True)
        r = run(LetheAdapter(embedder=embedder, vector_dim=384), "lethe")
        runs.append(r)
        _print_summary(r)

    if "mem0" in args.adapters:
        from bench.forgeteval.adapter import Mem0Adapter
        print("\nrunning mem0...", flush=True)
        r = run(Mem0Adapter(embedder_model=embed_model, embedding_dims=384),
                "mem0")
        runs.append(r)
        _print_summary(r)

    if "mempalace" in args.adapters:
        from bench.forgeteval.adapter import MemPalaceAdapter
        print("\nrunning mempalace...", flush=True)
        r = run(MemPalaceAdapter(), "mempalace")
        runs.append(r)
        _print_summary(r)

    if "langmem" in args.adapters:
        from bench.forgeteval.adapter import LangGraphAdapter
        print("\nrunning langmem (LangGraph InMemoryStore)...", flush=True)
        r = run(LangGraphAdapter(embedder=embedder, vector_dim=384), "langmem")
        runs.append(r)
        _print_summary(r)

    out = OUT / args.output
    out.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

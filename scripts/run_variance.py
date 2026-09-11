"""Variance / sensitivity sweep for the ForgetEval paper.

Runs the Lethe adapter on the generated benchmark across multiple seeds and
multiple distractor densities, dumping per-family pass rates to JSON.
Output lands in ../data/.

Usage:
    py scripts/run_variance.py            # full sweep, ~10 min
    py scripts/run_variance.py --quick    # smoke check, ~30s
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

LETHE_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LETHE_REPO))

OUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR.mkdir(exist_ok=True)


def _embedder_for(model_name: str):
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name)
    def embed(text: str) -> list[float]:
        return list(next(iter(model.embed([text]))))
    return embed


def run_one(*, seed: int, scale: int, distractors: int,
            adapter_name: str = "lethe",
            embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
            embedder=None, lang: str = "en") -> dict:
    from bench.forgeteval.generate import generate
    from bench.forgeteval.run import run_adapter

    if adapter_name == "lethe":
        from bench.forgeteval.adapter import LetheAdapter
        adapter = LetheAdapter(embedder=embedder, vector_dim=384)
    else:
        raise ValueError(adapter_name)

    cases = generate(scale, seed=seed, distractors=distractors, lang=lang)
    t0 = time.perf_counter()
    summary = run_adapter(adapter, cases, verbose=False)
    wall = time.perf_counter() - t0

    by_family: dict[str, dict] = {}
    total_pass = total = 0
    for fam, rows in summary["by_family"].items():
        p = sum(1 for _, ok, _ in rows if ok)
        by_family[fam] = {"pass": p, "total": len(rows),
                          "rate": p / max(len(rows), 1)}
        total_pass += p
        total += len(rows)

    return {
        "adapter": adapter_name,
        "lang": lang,
        "seed": seed,
        "scale_per_family": scale,
        "distractors": distractors,
        "embedder": embedder_model,
        "by_family": by_family,
        "overall_pass": total_pass,
        "overall_total": total,
        "overall_rate": total_pass / max(total, 1),
        "wall_seconds": wall,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    embedder_model = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"loading embedder: {embedder_model}", flush=True)
    embed = _embedder_for(embedder_model)

    if args.quick:
        seeds = [42, 43]
        scales = [10]
        distractors_list = [4]
    else:
        seeds = [42, 43, 44, 45, 46]
        scales = [50]
        distractors_list = [4]

    rows = []
    for scale in scales:
        for dis in distractors_list:
            for seed in seeds:
                tag = f"variance__seed{seed}_scale{scale}_d{dis}"
                print(f"  running {tag} ...", flush=True)
                r = run_one(seed=seed, scale=scale, distractors=dis,
                            embedder=embed, embedder_model=embedder_model)
                r["tag"] = tag
                rows.append(r)
                print(f"    overall = {r['overall_rate']:.4f} "
                      f"({r['overall_pass']}/{r['overall_total']}) "
                      f"wall={r['wall_seconds']:.1f}s",
                      flush=True)

    out = OUT_DIR / ("variance_quick.json" if args.quick else "variance.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {len(rows)} runs → {out}")


if __name__ == "__main__":
    main()

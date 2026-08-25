"""Distractor-density sweep. Tests robustness to background corpus growth.

For each distractor count d in {4, 10, 25, 50, 100}, run scale=50 seeds 42-44.
Output: ../data/distractors.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

LETHE_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LETHE_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_variance import OUT_DIR, _embedder_for, run_one


def main() -> None:
    embedder_model = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"loading embedder: {embedder_model}", flush=True)
    embed = _embedder_for(embedder_model)

    seeds = [42, 43, 44]
    scale = 50
    distractor_levels = [4, 10, 25, 50, 100]

    rows = []
    for d in distractor_levels:
        for seed in seeds:
            tag = f"distractors__d{d}_seed{seed}"
            print(f"  running {tag} ...", flush=True)
            r = run_one(seed=seed, scale=scale, distractors=d,
                        embedder=embed, embedder_model=embedder_model)
            r["tag"] = tag
            rows.append(r)
            print(f"    overall = {r['overall_rate']:.4f} "
                  f"({r['overall_pass']}/{r['overall_total']}) "
                  f"wall={r['wall_seconds']:.1f}s",
                  flush=True)

    out = OUT_DIR / "distractors.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {len(rows)} runs → {out}")


if __name__ == "__main__":
    main()

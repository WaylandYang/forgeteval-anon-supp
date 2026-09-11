"""Run the full ForgetEval at scale=200 (1000 cases) for the headline number.
Two seeds to confirm reproducibility of the 99.3% figure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

LETHE_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LETHE_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_variance import OUT_DIR, _embedder_for, run_one


def main():
    embed_model = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"loading embedder: {embed_model}", flush=True)
    embed = _embedder_for(embed_model)

    rows = []
    for seed in [42, 43]:
        tag = f"full__seed{seed}_scale200_d4"
        print(f"  running {tag} ...", flush=True)
        r = run_one(seed=seed, scale=200, distractors=4,
                    embedder=embed, embedder_model=embed_model)
        r["tag"] = tag
        rows.append(r)
        print(f"    overall = {r['overall_rate']:.4f} "
              f"({r['overall_pass']}/{r['overall_total']}) "
              f"wall={r['wall_seconds']:.1f}s", flush=True)

    out = OUT_DIR / "full_1000.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {len(rows)} runs → {out}")


if __name__ == "__main__":
    main()

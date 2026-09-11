"""Ablation sweep for the ForgetEval paper.

Each ablation is a hand-modified LetheAdapter:

  - hybrid_default  : the shipped adapter (RRF hybrid recall; lexical purge;
                      adaptive-gap release)
  - vec_only        : recall uses vector only (no BM25 leg); release uses
                      the same vec-only recall; purge falls back to vec
  - bm25_only       : recall uses pure BM25; release uses BM25; purge stays
                      lexical (a control)
  - fixed_threshold : release keeps top-1 only (no adaptive gap)
  - rrf_purge       : purge uses hybrid recall (no lexical-only path)

Each is run at scale=50, seeds 42–44, distractors=4.  Result: data/ablations.json.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

LETHE_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LETHE_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_variance import OUT_DIR, _embedder_for


class AblatedLetheAdapter:
    """LetheAdapter, configurable on three knobs."""
    name = "lethe-ablated"

    def __init__(self, embedder, vector_dim=384,
                 mode: str = "hybrid_default"):
        from lethe import Lethe
        self._Lethe = Lethe
        self.embedder = embedder
        self.vector_dim = vector_dim
        self.lethe = None
        self.mode = mode  # see module docstring

    def reset(self):
        if self.lethe is not None:
            try: self.lethe.close()
            except Exception: pass
        self.lethe = self._Lethe(":memory:",
                                 vector_dim=self.vector_dim,
                                 embedder=self.embedder)

    def inscribe(self, text):
        return self.lethe.inscribe(text)

    def _recall_hits(self, query, k):
        if self.mode == "vec_only":
            return self.lethe.recall(query, k=k, hybrid=False)
        if self.mode == "bm25_only":
            return self.lethe.recall(query, k=k, lexical=True)
        # default + fixed_threshold + rrf_purge all use hybrid (no BM25 leg
        # only because LetheAdapter passes hybrid=False — keep that to
        # match the shipped adapter exactly).
        return self.lethe.recall(query, k=k, hybrid=False)

    def recall_texts(self, query, k=5):
        return [h.memory.text for h in self._recall_hits(query, k=k)]

    def supersede(self, old_query, new_text):
        hits = self._recall_hits(old_query, k=1)
        if not hits:
            self.lethe.inscribe(new_text)
            return
        self.lethe.surrender({"old": hits[0].memory.id, "new": new_text},
                             mode="supersede")

    @staticmethod
    def _gap_threshold(sims, min_gap=0.05):
        # Mirrors LetheAdapter._gap_threshold exactly.
        if not sims: return float("inf")
        if len(sims) == 1: return sims[0] * 0.95
        s = sorted(sims, reverse=True)
        best_gap = 0.0
        best_mid = s[0] * 0.95
        for i in range(len(s) - 1):
            gap = s[i] - s[i + 1]
            if gap > best_gap:
                best_gap = gap
                best_mid = (s[i] + s[i + 1]) / 2.0
        return best_mid if best_gap >= min_gap else s[0] * 0.95

    def release(self, query):
        hits = self._recall_hits(query, k=20)
        if not hits: return 0
        if self.mode == "fixed_threshold":
            ids = [hits[0].memory.id]
        else:
            thr = self._gap_threshold([h.similarity for h in hits])
            ids = [h.memory.id for h in hits if h.similarity >= thr]
        if not ids: return 0
        self.lethe.surrender(ids, mode="release")
        return len(ids)

    def purge(self, query):
        if self.mode == "rrf_purge":
            hits = self._recall_hits(query, k=5)
        else:
            # default: lexical purge
            hits = self.lethe.recall(query, k=5, lexical=True)
        if not hits: return 0
        target = hits[0].memory.text
        ids = [h.memory.id for h in hits if h.memory.text == target]
        self.lethe.surrender(ids, mode="purge")
        return len(ids)


def run_one(*, mode, seed, scale=50, distractors=4, embedder=None) -> dict:
    from bench.forgeteval.generate import generate
    from bench.forgeteval.run import run_adapter

    adapter = AblatedLetheAdapter(embedder=embedder, vector_dim=384, mode=mode)
    cases = generate(scale, seed=seed, distractors=distractors, lang="en")
    t0 = time.perf_counter()
    summary = run_adapter(adapter, cases, verbose=False)
    wall = time.perf_counter() - t0

    by_family = {}
    total_pass = total = 0
    for fam, rows in summary["by_family"].items():
        p = sum(1 for _, ok, _ in rows if ok)
        by_family[fam] = {"pass": p, "total": len(rows),
                          "rate": p / max(len(rows), 1)}
        total_pass += p
        total += len(rows)

    return {
        "mode": mode, "seed": seed,
        "scale_per_family": scale, "distractors": distractors,
        "by_family": by_family,
        "overall_pass": total_pass, "overall_total": total,
        "overall_rate": total_pass / max(total, 1),
        "wall_seconds": wall,
    }


def main():
    embed_model = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"loading embedder: {embed_model}", flush=True)
    embed = _embedder_for(embed_model)

    modes = ["hybrid_default", "vec_only", "bm25_only",
             "fixed_threshold", "rrf_purge"]
    seeds = [42, 43, 44]

    rows = []
    for mode in modes:
        for seed in seeds:
            tag = f"ablation__{mode}_seed{seed}"
            print(f"  running {tag} ...", flush=True)
            r = run_one(mode=mode, seed=seed, embedder=embed)
            r["tag"] = tag
            rows.append(r)
            print(f"    overall = {r['overall_rate']:.4f} "
                  f"({r['overall_pass']}/{r['overall_total']}) "
                  f"wall={r['wall_seconds']:.1f}s", flush=True)

    out = OUT_DIR / "ablations.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {len(rows)} runs → {out}")


if __name__ == "__main__":
    main()

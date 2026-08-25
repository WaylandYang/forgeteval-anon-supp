"""Cross-LLM ablation: run Lethe+LLM on the adversarial bench with
different LLMs wired into the supersede / purge_match / release_match
hooks.  Addresses the "single LLM evaluated" limitation.

Models tested (all via SiliconFlow, non-thinking variants):
  - deepseek-ai/DeepSeek-V3      (the headline result)
  - Qwen/Qwen2.5-72B-Instruct    (cross-family check, judge model)
  - meta-llama/Meta-Llama-3.1-70B-Instruct  (third family)

Output:
    data/cross_llm_ablation.json   per-model per-category scores

Usage:
    py scripts/run_cross_llm_ablation.py [--models m1 m2 ...]
                                          [--limit N]   # debug
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

LETHE_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LETHE_REPO))

SF_KEY = os.environ.get("SILICONFLOW_API_KEY") or \
    ""
SF_BASE = "https://api.siliconflow.cn/v1"

OUT = Path(__file__).resolve().parent.parent / "data" / "cross_llm_ablation.json"

DEFAULT_MODELS = [
    "deepseek-ai/DeepSeek-V3",
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
]


def make_llm(model: str):
    import openai
    client = openai.OpenAI(api_key=SF_KEY, base_url=SF_BASE)

    def llm(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=2048,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""
    return llm


def build_lethe_with_llm(llm):
    from bench.forgeteval.adapter import LetheAdapter
    from fastembed import TextEmbedding
    model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    def embedder(t):
        return list(next(iter(model.embed([t]))))
    return LetheAdapter(embedder=embedder, vector_dim=384, llm=llm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit total cases (debug).")
    args = ap.parse_args()

    from bench.forgeteval.adversarial import ATTACK_CATEGORIES
    all_cases = []
    for cat, cases in ATTACK_CATEGORIES.items():
        for c in cases:
            all_cases.append((cat, c))
    if args.limit:
        all_cases = all_cases[: args.limit]
    print(f"Cases: {len(all_cases)} across "
          f"{len(ATTACK_CATEGORIES)} categories\n")

    all_runs = []
    for model in args.models:
        print(f"\n========== {model} ==========")
        llm = make_llm(model)
        adapter = build_lethe_with_llm(llm)
        t0 = time.perf_counter()
        per_case = []
        n_pass = 0
        by_cat: dict = {}
        for i, (cat, c) in enumerate(all_cases, 1):
            try:
                passed = c.run(adapter)
            except Exception as e:
                passed = False
                err = f"{type(e).__name__}: {e}"
            else:
                err = None
            per_case.append({
                "case_id": c.id, "category": cat,
                "passed": passed, "error": err,
            })
            d = by_cat.setdefault(cat, {"total": 0, "pass": 0})
            d["total"] += 1
            if passed:
                d["pass"] += 1
                n_pass += 1
            if i % 20 == 0 or i == len(all_cases):
                print(f"  [{i:3}/{len(all_cases)}] "
                      f"{n_pass}/{i} = {n_pass/i*100:.1f}%",
                      flush=True)
        wall = time.perf_counter() - t0
        print(f"  DONE: {n_pass}/{len(all_cases)} = "
              f"{n_pass/len(all_cases)*100:.1f}% in {wall:.1f}s")
        for cat in ATTACK_CATEGORIES:
            d = by_cat.get(cat, {"total": 0, "pass": 0})
            if d["total"]:
                print(f"    {cat:<28} {d['pass']}/{d['total']} "
                      f"({d['pass']/d['total']*100:.0f}%)")
        all_runs.append({
            "model": model,
            "overall_pass": n_pass,
            "overall_total": len(all_cases),
            "by_category": by_cat,
            "wall_seconds": wall,
            "per_case": per_case,
        })
        # Incremental save.
        OUT.write_text(json.dumps(all_runs, ensure_ascii=False, indent=2),
                       encoding="utf-8")

    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()


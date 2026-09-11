"""Run ForgetEval-Adv with a SiliconFlow LLM wired into LetheAdapter.

SiliconFlow exposes OpenAI-compatible endpoints with many open-weight
models — we pick a fast non-thinking instruct model so the JSON-shaped
prompts complete in one short response per call.

Default model:  Qwen/Qwen2.5-7B-Instruct
Why this model:
  - Cheap (~$0.0001 per 1k input tokens on SiliconFlow as of 2026)
  - Fast (small enough that per-call latency is < 1 s)
  - NOT a thinking variant — returns the JSON directly, no
    <think>...</think> wrappers
  - Strong enough instruction-following for our narrow JSON contract

The user can override via LETHE_LLM_MODEL=<model_id>.

Output: ../data/adversarial_results_with_llm_siliconflow.json
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

MODEL = os.environ.get("LETHE_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
SF_KEY = os.environ.get("SILICONFLOW_API_KEY") or \
    ""
SF_BASE = "https://api.siliconflow.cn/v1"


def make_siliconflow_llm():
    """Return a Callable[[str], str] that proxies prompt→response through
    SiliconFlow's OpenAI-compatible chat completions API.  Caches
    results in-memory so retries don't double-bill."""
    import openai
    client = openai.OpenAI(api_key=SF_KEY, base_url=SF_BASE)
    cache: dict[str, str] = {}
    calls = {"hits": 0, "misses": 0, "errors": 0}

    def llm(prompt: str) -> str:
        if prompt in cache:
            calls["hits"] += 1
            return cache[prompt]
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=512,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content or ""
            cache[prompt] = text
            calls["misses"] += 1
            return text
        except Exception as e:
            calls["errors"] += 1
            print(f"  [llm error] {type(e).__name__}: {e}", flush=True)
            return ""

    llm.calls = calls           # type: ignore
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
        mark = "pass" if passed else "fail"
        print(f"  [{i:>3}/{len(ADVERSARIAL_TESTS)}] {mark:<4} {case.id}",
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
        "provider": "siliconflow",
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

    print(f"using LLM model:  {MODEL}  (via SiliconFlow)", flush=True)
    llm = make_siliconflow_llm()
    adapter = LetheAdapter(embedder=embedder, vector_dim=384, llm=llm)

    print(f"\nrunning lethe + llm on adversarial (112 cases)...\n",
          flush=True)
    r = run(adapter, "lethe+llm")

    calls = getattr(llm, "calls", {})
    print(f"\noverall = {r['overall_rate']:.4f} "
          f"({r['overall_pass']}/{r['overall_total']})  "
          f"wall={r['wall_seconds']:.1f}s  "
          f"llm: {calls.get('misses', 0)} calls, "
          f"{calls.get('hits', 0)} cache hits, "
          f"{calls.get('errors', 0)} errors\n", flush=True)
    print("by attack category:")
    for cat, d in r["by_attack_category"].items():
        print(f"  {cat:<28} {d['pass']:>2}/{d['total']:<2} ({d['rate']:.0%})",
              flush=True)

    out = OUT / "adversarial_results_with_llm_siliconflow.json"
    out.write_text(json.dumps([r], indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()


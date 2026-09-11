"""Run LangGraph + DeepSeek-V3 LLM hook on ForgetEval-Adv.

Disentangles "LLM hook" from "edit primitive" by giving LangGraph the
same LLM hook as Lethe+LLM.

Output: data/adversarial_results_v05_langgraph_llm.json
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

LETHE_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LETHE_REPO))

OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)

SF_KEY = os.environ.get("SILICONFLOW_API_KEY") or \
    ""
SF_BASE = "https://api.siliconflow.cn/v1"
MODEL = os.environ.get("LETHE_LLM_MODEL", "deepseek-ai/DeepSeek-V3")


def make_llm():
    import openai
    client = openai.OpenAI(api_key=SF_KEY, base_url=SF_BASE)
    cache: dict[str, str] = {}
    n_calls = [0]
    n_cache = [0]

    def llm(prompt: str) -> str:
        if prompt in cache:
            n_cache[0] += 1
            return cache[prompt]
        n_calls[0] += 1
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=2048,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        out = resp.choices[0].message.content or ""
        cache[prompt] = out
        return out

    llm._counters = (n_calls, n_cache)
    return llm


def main():
    from bench.forgeteval.adversarial import (
        ADVERSARIAL_TESTS, ATTACK_CATEGORIES, case_to_attack_category,
    )
    from bench.forgeteval.adapter import LangGraphLLMAdapter
    from fastembed import TextEmbedding

    print(f"loading embedder...", flush=True)
    model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    def embedder(t):
        return list(next(iter(model.embed([t]))))

    print(f"using LLM: {MODEL} (SiliconFlow)", flush=True)
    llm = make_llm()
    adapter = LangGraphLLMAdapter(embedder=embedder, vector_dim=384, llm=llm)

    print(f"\nrunning langmem+llm on {len(ADVERSARIAL_TESTS)} cases...\n",
          flush=True)
    results = []
    n_pass = 0
    t0 = time.perf_counter()
    for i, case in enumerate(ADVERSARIAL_TESTS, 1):
        cat = case_to_attack_category(case.id)
        try:
            passed = case.run(adapter)
            err = None
        except NotImplementedError:
            passed = False
            err = "N/A"
        except Exception as e:
            passed = False
            err = f"{type(e).__name__}: {e}"
        results.append({
            "id": case.id, "family": case.family,
            "attack_category": cat, "passed": passed, "error": err,
        })
        if passed:
            n_pass += 1
        if i % 20 == 0 or i == len(ADVERSARIAL_TESTS):
            print(f"  [{i:3}/{len(ADVERSARIAL_TESTS)}] "
                  f"{'pass' if passed else 'fail'} {case.id}  "
                  f"so-far {n_pass}/{i} = {n_pass/i*100:.1f}%",
                  flush=True)

    wall = time.perf_counter() - t0
    n_total = len(results)
    n_calls, n_cache = llm._counters

    print(f"\noverall = {n_pass/n_total:.4f} ({n_pass}/{n_total})  "
          f"wall={wall:.1f}s  llm: {n_calls[0]} calls, "
          f"{n_cache[0]} cache hits, "
          f"{sum(1 for r in results if r['error'])} errors")

    # Per-category aggregation
    by_cat: dict[str, dict] = {}
    for cat in ATTACK_CATEGORIES:
        rows = [r for r in results if r["attack_category"] == cat]
        passed = sum(1 for r in rows if r["passed"])
        by_cat[cat] = {"pass": passed, "total": len(rows),
                        "rate": passed / max(len(rows), 1)}
        if rows:
            print(f"  {cat:<28} {passed:>2}/{len(rows):<3} "
                  f"({passed/len(rows)*100:>3.0f}%)")

    by_family: dict[str, dict] = defaultdict(lambda: {"pass": 0, "total": 0})
    for r in results:
        by_family[r["family"]]["total"] += 1
        if r["passed"]:
            by_family[r["family"]]["pass"] += 1
    for fam, d in by_family.items():
        d["rate"] = d["pass"] / max(d["total"], 1)

    out_data = [{
        "adapter": "langmem_llm",
        "model": MODEL,
        "suite": "adversarial",
        "case_count": n_total,
        "overall_pass": n_pass,
        "overall_total": n_total,
        "overall_rate": n_pass / max(n_total, 1),
        "by_family": dict(by_family),
        "by_attack_category": by_cat,
        "per_case": results,
        "wall_seconds": wall,
        "llm_calls": n_calls[0],
        "llm_cache_hits": n_cache[0],
    }]
    out_name = os.environ.get(
        "LANGGRAPH_LLM_OUT", "adversarial_results_v05_langgraph_llm.json"
    )
    out_path = OUT / out_name
    out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()


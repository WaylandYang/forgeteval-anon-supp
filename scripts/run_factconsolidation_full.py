"""Extended FactConsolidation cross-evaluation on MemoryAgentBench
(ICLR 2026) Conflict Resolution competency.

Scope: 4 context-length buckets (sh_6k, mh_6k, sh_32k, mh_32k) ×
full 100 questions each = 400 questions, across 6 systems:
deterministic (Lethe, LangGraph, MemPalace, Mem0) + LLM-hook
(Lethe+LLM, LangGraph+LLM).

Purpose: independent third-party validation of (a) "FactConsolidation
under substring scoring is recall-shaped" claim from Appendix H and
(b) axis-flip: LLM-hook variants help forgetting (ForgetEval-Adv)
but should NOT lift recall (FactConsolidation), since no supersede
/ release / purge primitive is invoked in this pure-recall task.

Skips 64k and 262k buckets: their >270k-char contexts blow our
single-CPU adapter's in-memory index budget.

Output: data/factconsolidation_full.json
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "lethe"))

from bench.forgeteval.adapter import (  # noqa: E402
    LetheAdapter, LangGraphAdapter, MemPalaceAdapter, Mem0Adapter,
    LangGraphLLMAdapter,
)
from datasets import load_dataset  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)

SF_KEY = os.environ.get("SILICONFLOW_API_KEY") or \
    ""
SF_BASE = "https://api.siliconflow.cn/v1"
HOOK_MODEL = "deepseek-ai/DeepSeek-V3"

BUCKETS = ("factconsolidation_sh_6k",
           "factconsolidation_mh_6k",
           "factconsolidation_sh_32k",
           "factconsolidation_mh_32k")


def parse_context(ctx: str) -> list[str]:
    out = []
    for line in ctx.splitlines():
        m = re.match(r"^\s*\d+\.\s+(.+)$", line.strip())
        if m:
            out.append(m.group(1).strip())
    return out


def scorer(retrieved_texts: list[str], gold_answers: list[str]) -> bool:
    blob = " ".join(retrieved_texts).lower()
    return any(g.lower() in blob for g in gold_answers)


def evaluate(adapter, facts, questions, golds, k=10):
    adapter.reset()
    t0 = time.time()
    for f in facts:
        adapter.inscribe(f)
    inscribe_s = time.time() - t0
    retrievals = []
    t0 = time.time()
    for q in questions:
        retrievals.append(adapter.recall_texts(q, k=k))
    recall_s = time.time() - t0
    pass_k1 = pass_k5 = pass_k10 = 0
    per_q = []
    for q, g, r in zip(questions, golds, retrievals):
        ok1 = scorer(r[:1], g)
        ok5 = scorer(r[:5], g)
        ok10 = scorer(r[:k], g)
        pass_k1 += int(ok1); pass_k5 += int(ok5); pass_k10 += int(ok10)
        per_q.append({"q": q, "gold": g,
                      "pass_k1": ok1, "pass_k5": ok5, "pass_k10": ok10})
    n = len(questions)
    return {"n": n,
            "pass_k1": pass_k1, "rate_k1": pass_k1 / n,
            "pass_k5": pass_k5, "rate_k5": pass_k5 / n,
            "pass_k10": pass_k10, "rate_k10": pass_k10 / n,
            "inscribe_s": inscribe_s, "recall_s": recall_s,
            "per_question": per_q}


def load_bucket(ds, bucket: str):
    """Return (facts, questions, golds) for one bucket row."""
    for row in ds:
        qa = row["metadata"].get("qa_pair_ids", []) or []
        if qa and qa[0].rsplit("_no", 1)[0] == bucket:
            facts = parse_context(row["context"])
            return facts, row["questions"], row["answers"]
    raise ValueError(f"bucket {bucket} not found")


def main():
    print("loading MemoryAgentBench Conflict_Resolution...", flush=True)
    ds = load_dataset("ai-hyz/MemoryAgentBench")["Conflict_Resolution"]

    print("loading embedder (all-MiniLM-L6-v2)...", flush=True)
    from fastembed import TextEmbedding
    model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    def embedder(t):
        return list(next(iter(model.embed([t]))))

    def make_llm():
        import openai
        client = openai.OpenAI(api_key=SF_KEY, base_url=SF_BASE)
        def llm(prompt: str) -> str:
            resp = client.chat.completions.create(
                model=HOOK_MODEL, max_tokens=2048, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""
        return llm

    factories = [
        ("Lethe",          lambda: LetheAdapter(embedder=embedder, vector_dim=384)),
        ("LangGraph",      lambda: LangGraphAdapter(embedder=embedder, vector_dim=384)),
        ("MemPalace",      lambda: MemPalaceAdapter()),
        ("Mem0",           lambda: Mem0Adapter()),
        ("Lethe+LLM",      lambda: LetheAdapter(embedder=embedder, vector_dim=384,
                                                 llm=make_llm())),
        ("LangGraph+LLM",  lambda: LangGraphLLMAdapter(embedder=embedder, vector_dim=384,
                                                       llm=make_llm())),
    ]

    out_blocks = {}
    for bucket in BUCKETS:
        print(f"\n=== {bucket} ===", flush=True)
        try:
            facts, questions, golds = load_bucket(ds, bucket)
        except Exception as e:
            print(f"  skip ({e})"); continue
        print(f"  {len(facts)} facts, {len(questions)} questions")
        results = {}
        for name, factory in factories:
            try:
                a = factory()
            except Exception as e:
                print(f"  {name}: factory failed: {type(e).__name__}: {e}")
                continue
            t0 = time.time()
            try:
                r = evaluate(a, facts, questions, golds)
            except Exception as e:
                print(f"  {name}: eval failed: {type(e).__name__}: {e}")
                continue
            print(f"  {name:14s}  k=10: {r['pass_k10']:3d}/{r['n']:<3d} "
                  f"({r['rate_k10']*100:5.1f}%)  k=5: {r['pass_k5']:3d}/{r['n']:<3d}  "
                  f"k=1: {r['pass_k1']:3d}/{r['n']:<3d}  "
                  f"wall={time.time()-t0:.1f}s", flush=True)
            results[name] = r
        out_blocks[bucket] = {"n_facts": len(facts),
                              "n_questions": len(questions),
                              "results": results}
        # Incremental save
        out_path = OUT / "factconsolidation_full.json"
        payload = {
            "scope": "MemoryAgentBench Conflict_Resolution: sh+mh × 6k+32k buckets, "
                     "full 100 questions per bucket, 6 systems",
            "scoring": "case-insensitive substring of gold in top-k retrieval",
            "blocks": {b: {"n_facts": v["n_facts"], "n_questions": v["n_questions"],
                            "results": {n: {k: v_ for k, v_ in r.items() if k != "per_question"}
                                         for n, r in v["results"].items()}}
                       for b, v in out_blocks.items()},
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"  wrote {out_path}", flush=True)

    print("\n=== Summary (top-10 substring rate) ===", flush=True)
    headers = ["System"] + list(BUCKETS)
    print(" | ".join(h[:14].ljust(14) for h in headers))
    for name, _ in factories:
        row = [name]
        for b in BUCKETS:
            r = out_blocks.get(b, {}).get("results", {}).get(name)
            row.append(f"{r['pass_k10']}/{r['n']} ({r['rate_k10']*100:.0f}%)" if r else "-")
        print(" | ".join(c[:14].ljust(14) for c in row))


if __name__ == "__main__":
    main()


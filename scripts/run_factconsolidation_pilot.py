"""Mini FactConsolidation cross-eval pilot.

Cross-evaluation on an external benchmark: run our adapters against
MemoryAgentBench's
Conflict_Resolution (FactConsolidation) split.

Scope: factconsolidation_sh_6k single-hop, first 50 QA pairs.
Scoring: deterministic substring of gold answer in concatenated top-10 retrieval.
Systems: Lethe (det), LangGraph (det), MemPalace.

Multi-hop (mh) is out of scope: no retrieval-only system answers multi-hop chained
queries without an LLM reasoning step, and our adversarial layer is the place we
make the architecture-agnostic LLM-hook claim.  We report sh only and note this.
"""
from __future__ import annotations
import json
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# Allow importing the adapter module from the lethe repo
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "lethe"))

from bench.forgeteval.adapter import (  # noqa: E402
    LetheAdapter,
    LangGraphAdapter,
    MemPalaceAdapter,
)

from datasets import load_dataset  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)


def parse_context(ctx: str) -> list[str]:
    """Each line is 'N. <fact>'.  Skip the lead-in 'Here is a list of facts:'."""
    out = []
    for line in ctx.splitlines():
        m = re.match(r"^\s*\d+\.\s+(.+)$", line.strip())
        if m:
            out.append(m.group(1).strip())
    return out


def scorer(retrieved_texts: list[str], gold_answers: list[str]) -> bool:
    """Pass iff any gold answer is a case-insensitive substring of joined recall."""
    blob = " ".join(retrieved_texts).lower()
    return any(g.lower() in blob for g in gold_answers)


def evaluate(adapter, facts: list[str], questions: list[str],
             golds: list[list[str]]) -> dict:
    adapter.reset()
    t0 = time.time()
    for f in facts:
        adapter.inscribe(f)
    inscribe_s = time.time() - t0

    # Always retrieve k=10 once; score at k=1 (strict: edit must win retrieval),
    # k=5, and k=10 (lenient: edit must be in top-10).
    retrievals = []
    t0 = time.time()
    for q in questions:
        retrievals.append(adapter.recall_texts(q, k=10))
    recall_s = time.time() - t0

    per_q = []
    pass_k1 = pass_k5 = pass_k10 = 0
    for q, g, r in zip(questions, golds, retrievals):
        ok1 = scorer(r[:1], g)
        ok5 = scorer(r[:5], g)
        ok10 = scorer(r[:10], g)
        pass_k1 += int(ok1)
        pass_k5 += int(ok5)
        pass_k10 += int(ok10)
        per_q.append({"q": q, "gold": g, "top1": r[:1],
                      "pass_k1": ok1, "pass_k5": ok5, "pass_k10": ok10})

    n = len(questions)
    return {
        "n": n,
        "pass_k1": pass_k1, "rate_k1": pass_k1 / n,
        "pass_k5": pass_k5, "rate_k5": pass_k5 / n,
        "pass_k10": pass_k10, "rate_k10": pass_k10 / n,
        "inscribe_s": inscribe_s,
        "recall_s": recall_s,
        "per_question": per_q,
    }


def run_split(source: str, n: int, adapters_factory) -> dict:
    """Load one row of Conflict_Resolution, run all adapters on first n questions."""
    ds = load_dataset("ai-hyz/MemoryAgentBench", split="Conflict_Resolution")
    row = next(r for r in ds if r["metadata"]["source"] == source)
    facts = parse_context(row["context"])
    print(f"[{source}] {len(facts)} facts, {n} questions")

    questions = row["questions"][:n]
    golds = row["answers"][:n]

    results = {}
    for name, factory in adapters_factory:
        a = factory()
        print(f"  --- {name} ---")
        t0 = time.time()
        r = evaluate(a, facts, questions, golds)
        print(f"    k=10: {r['pass_k10']}/{r['n']} = {r['rate_k10']*100:.1f}%  "
              f"(inscribe {r['inscribe_s']:.1f}s, recall {r['recall_s']:.1f}s, "
              f"total {time.time()-t0:.1f}s)")
        results[name] = r
    return {"source": source, "n_facts": len(facts), "n_questions": n,
            "results": results}


def main():
    N = 50

    print("loading embedder (all-MiniLM-L6-v2)...")
    from fastembed import TextEmbedding
    model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

    def embedder(text: str) -> list[float]:
        return list(next(iter(model.embed([text]))))

    factories = [
        ("Lethe",     lambda: LetheAdapter(embedder=embedder, vector_dim=384)),
        ("LangGraph", lambda: LangGraphAdapter(embedder=embedder, vector_dim=384)),
        ("MemPalace", lambda: MemPalaceAdapter()),
    ]

    out_blocks = {}
    for src in ("factconsolidation_sh_6k", "factconsolidation_mh_6k"):
        out_blocks[src] = run_split(src, N, factories)

    out_path = OUT / "factconsolidation_pilot_n50.json"
    payload = {
        "scope": "Conflict_Resolution sh_6k + mh_6k, first 50 QA pairs each",
        "scoring": "case-insensitive substring of gold in top-k retrieval (k=1,5,10)",
        "note_on_mempalace": "MemPalace.recall_texts returns one concatenated blob, so k=1 is not apples-to-apples across systems; treat k=10 as the headline.",
        "blocks": {src: {"source": src,
                          "n_facts": b["n_facts"],
                          "n_questions": b["n_questions"],
                          "results": {n: {k: v for k, v in r.items() if k != "per_question"}
                                       for n, r in b["results"].items()},
                          "per_question": {n: r["per_question"]
                                            for n, r in b["results"].items()}}
                    for src, b in out_blocks.items()},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}")

    print("\n=== Summary (top-10 substring) ===")
    print(f"{'System':12s}  {'sh_6k k=10':>12s}  {'mh_6k k=10':>12s}")
    for name, _ in factories:
        sh = out_blocks["factconsolidation_sh_6k"]["results"][name]
        mh = out_blocks["factconsolidation_mh_6k"]["results"][name]
        print(f"{name:12s}  {sh['pass_k10']:3d}/{sh['n']:<3d}  "
              f"     {mh['pass_k10']:3d}/{mh['n']:<3d}")


if __name__ == "__main__":
    main()

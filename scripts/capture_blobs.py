"""Capture per-case retrieved blobs for a system on the full 385-case suite.

Output feeds the NLI-aware scorer (scripts/nli_scorer.py): each record has the
case spec + the actual top-10 retrieval, plus the deterministic substring
verdict so we can diff substring-vs-NLI.

  python scripts/capture_blobs.py --adapter lethe          # deterministic, local, ~18s
  OPENROUTER_API_KEY=.. OPENROUTER_MODEL=deepseek/deepseek-v4-pro \
      python scripts/capture_blobs.py --adapter lethe+llm  # hooked, via OpenRouter
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

LETHE_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LETHE_REPO))
OUT = Path(__file__).resolve().parent.parent / "data"

from bench.forgeteval.adversarial import (  # noqa: E402
    ADVERSARIAL_TESTS, case_to_attack_category,
)
from bench.forgeteval.adapter import LetheAdapter  # noqa: E402


def substring_pass(case, blob):
    b = blob.lower()
    for s in case.must_contain:
        if s.lower() not in b:
            return False, f"missing must_contain {s!r}"
    for t in case.must_not_contain:
        if t.lower() in b:
            return False, f"leaked must_not_contain {t!r}"
    return True, "pass"


def build_adapter(kind):
    from fastembed import TextEmbedding
    m = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    emb = lambda t: list(next(iter(m.embed([t]))))
    if kind == "lethe":
        return LetheAdapter(embedder=emb, vector_dim=384)
    # lethe+llm: wire OpenRouter hook
    import openai
    key = os.environ.get("OPENROUTER_API_KEY")
    model = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-v4-pro")
    base = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    client = openai.OpenAI(api_key=key, base_url=base)
    cache = {}

    def llm(prompt):
        if prompt in cache:
            return cache[prompt]
        try:
            r = client.chat.completions.create(
                model=model,
                max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "3000")),
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}])
            t = r.choices[0].message.content or ""
            cache[prompt] = t
            return t
        except Exception as e:
            print(f"  [llm err] {str(e)[:80]}")
            return ""
    return LetheAdapter(embedder=emb, vector_dim=384, llm=llm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="lethe", choices=["lethe", "lethe+llm"])
    ap.add_argument("--suite", choices=["v051", "v07"], default="v051")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    adapter = build_adapter(args.adapter)
    if args.suite == "v07":
        from scripts.repair_cross_lingual_queries import build_suite
        SUITE, _ = build_suite()
    else:
        SUITE = ADVERSARIAL_TESTS
    slug = args.adapter.replace("+", "_") + \
        ("" if args.suite == "v051" else "_" + args.suite)
    ckpt = OUT / f"blobs_{slug}_ckpt.jsonl"
    done = set()
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
        print(f"resume: {len(done)} done")

    fout = ckpt.open("a", encoding="utf-8")
    npass = 0
    for i, c in enumerate(SUITE):
        if c.id in done:
            continue
        adapter.reset()
        for f in c.setup_facts:
            adapter.inscribe(f)
        for mm in c.mutations:
            op = mm[0]
            try:
                if op == "supersede":
                    adapter.supersede(mm[1], mm[2])
                elif op == "release":
                    adapter.release(mm[1])
                elif op == "purge":
                    adapter.purge(mm[1])
            except Exception as e:
                print(f"  [mut err] {c.id}: {str(e)[:60]}")
        top = adapter.recall_texts(c.final_query, k=10)
        probes = []
        for q in c.must_not_contain:
            if q and q.strip():
                probes.extend(adapter.recall_texts(q.strip(), 10))
        blob = " ".join(top + probes)
        ok, reason = substring_pass(c, blob)
        npass += ok
        rec = {"id": c.id, "category": case_to_attack_category(c.id),
               "final_query": c.final_query,
               "must_contain": c.must_contain,
               "must_not_contain": c.must_not_contain,
               "retrieved": top, "substring_pass": ok,
               "substring_reason": reason}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/385  substring_pass≈{npass}", flush=True)
    fout.close()

    # consolidate to a clean json
    records = [json.loads(l) for l in ckpt.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = OUT / f"blobs_{slug}.json"
    out.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    sp = sum(r["substring_pass"] for r in records)
    print(f"\n{args.adapter}: {len(records)} cases, substring {sp}/{len(records)} = {sp/len(records):.1%}")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()

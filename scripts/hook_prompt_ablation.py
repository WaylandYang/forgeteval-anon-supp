"""Hook prompt-robustness ablation.

Tests whether the mutation-time hook lift is a fragile prompt-engineering
artifact or a robust pattern, by swapping the SUPERSEDE prompt for three
variants and re-running the full 385 on a fixed model + backend:

  v0_original : the shipped 4-shot prompt
  v1_zeroshot : same instructions, EXAMPLES section removed
  v2_reworded : terse paraphrase, no examples, different wording

If the lift is stable across variants, the hook is robust to prompt
wording (not few-shot- or phrasing-dependent). Checkpointed/resumable.

  OPENROUTER_API_KEY=.. python scripts/hook_prompt_ablation.py --variant v1_zeroshot
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

LETHE_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LETHE_REPO))
OUT = LETHE_REPO / "data"

import bench.forgeteval.adapter as A  # noqa: E402
from bench.forgeteval.adversarial import (  # noqa: E402
    ADVERSARIAL_TESTS, case_to_attack_category,
)
from bench.forgeteval.adapter import LetheAdapter
from bench.forgeteval.scoring import run_scored  # noqa: E402

KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324")
BASE = "https://openrouter.ai/api/v1"

# v1: original minus the Examples block (everything before "Examples" + the Format tail)
_orig = A.LLM_PROMPT_SUPERSEDE
_pre = _orig.split("Examples")[0]
_format = "Format\n------\n" + _orig.split("Format\n------\n", 1)[1]
PROMPT_ZEROSHOT = _pre + _format

PROMPT_REWORDED = """\
Decide how to apply a memory update.

CURRENT:  {old_text}
TOPIC:    {query}
UPDATE:   {new_text}

If CURRENT is about a single topic that TOPIC names, replace it wholesale
(atomic) — even if the wording differs or there are dates or negations.
Only if CURRENT bundles two unrelated attributes about one subject and
TOPIC names just one of them, keep the other attribute and weave UPDATE
into the named one (partial).

Output one JSON object only:
  {{"mode": "atomic"}}
or {{"mode": "partial", "merged_text": "<merged sentence>"}}
Invent nothing not in CURRENT or UPDATE.
"""

VARIANTS = {
    "v0_original": _orig,
    "v1_zeroshot": PROMPT_ZEROSHOT,
    "v2_reworded": PROMPT_REWORDED,
}


def make_llm():
    import openai
    client = openai.OpenAI(api_key=KEY, base_url=BASE)
    cache = {}

    def llm(prompt):
        if prompt in cache:
            return cache[prompt]
        try:
            r = client.chat.completions.create(
                model=MODEL, max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "3000")), temperature=0.0,
                messages=[{"role": "user", "content": prompt}])
            t = r.choices[0].message.content or ""
            cache[prompt] = t
            return t
        except Exception as e:
            print(f"  [llm err] {str(e)[:70]}")
            return ""
    return llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--suite", choices=["v051", "v07"], default="v051")
    ap.add_argument("--probed", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if not KEY:
        sys.exit("set OPENROUTER_API_KEY")

    # swap the supersede prompt
    A.LLM_PROMPT_SUPERSEDE = VARIANTS[args.variant]
    print(f"variant {args.variant}: supersede prompt len={len(VARIANTS[args.variant])}")

    from fastembed import TextEmbedding
    em = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    emb = lambda t: list(next(iter(em.embed([t]))))
    adapter = LetheAdapter(embedder=emb, vector_dim=384, llm=make_llm())

    if args.suite == "v07":
        from scripts.repair_cross_lingual_queries import build_suite
        SUITE, _ = build_suite()
    else:
        SUITE = ADVERSARIAL_TESTS
    tag = ("" if args.suite == "v051" else "_" + args.suite) + \
          ("_probed" if args.probed else "")
    ckpt = OUT / f"ablate_{args.variant}{tag}_ckpt.jsonl"
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line); done[r["id"]] = r["ok"]
        print(f"resume: {len(done)} done")

    fout = ckpt.open("a", encoding="utf-8")
    bycat = defaultdict(lambda: {"p": 0, "t": 0})
    npass = 0
    t0 = time.perf_counter()
    for i, c in enumerate(SUITE):
        if c.id in done:
            ok = done[c.id]
        else:
            try:
                ok = (run_scored(c, adapter, probed=True) if args.probed
                      else c.run(adapter))
            except Exception as e:
                ok = False
                print(f"  [case err] {c.id}: {str(e)[:60]}")
            fout.write(json.dumps({"id": c.id, "ok": ok}) + "\n")
            fout.flush()
        cat = case_to_attack_category(c.id)
        bycat[cat]["t"] += 1
        if ok:
            bycat[cat]["p"] += 1; npass += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/385  pass={npass}", flush=True)
    fout.close()

    total = len(ADVERSARIAL_TESTS)
    print(f"\n=== {args.variant}: {npass}/{total} = {npass/total:.1%}  ({time.perf_counter()-t0:.0f}s) ===")
    for cat in sorted(bycat):
        d = bycat[cat]
        print(f"  {cat:<26} {d['p']}/{d['t']}")
    out = {"variant": args.variant, "model": MODEL, "overall_pass": npass,
           "overall_total": total, "overall_rate": npass / total,
           "by_category": {k: dict(v) for k, v in bycat.items()}}
    (OUT / f"ablate_{args.variant}{tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote ablate_{args.variant}.json")


if __name__ == "__main__":
    main()

"""NLI-aware re-scoring of a system's retrievals (reviewer lever b).

Substring scoring has two documented blind spots: a must_not_contain id that
is a substring of a legitimately-surviving longer id (prefix_collision), and
an entity that survives only inside a negated / past-tense clause.  This
re-scores each case SEMANTICALLY with an LLM judge and diffs substring-vs-NLI
per category, quantifying how much the substring scorer over/under-counts.

  python scripts/nli_scorer.py --blobs data/blobs_lethe.json [--limit N]

Reads OPENROUTER_API_KEY + NLI_MODEL (default a strong judge) from env.
Checkpointed: safe to kill/resume.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from textwrap import dedent

OUT = Path(__file__).resolve().parent.parent / "data"
KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = os.environ.get("NLI_MODEL", "openai/gpt-5.5")
BASE = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")

JUDGE = dedent("""\
You are a strict semantic judge for a memory-forgetting benchmark. A memory
system was asked a query and returned some retrieved text. Decide whether the
retrieval satisfies the requirements SEMANTICALLY (not by substring).

QUERY: {query}

RETRIEVED TEXT (top-10 memory rows, joined):
{blob}

REQUIREMENTS:
- MUST be semantically present (the fact is actually conveyed): {mc}
- MUST NOT be semantically present (the fact must be truly gone/forgotten): {mnc}

Judge SEMANTICALLY, correcting the known substring pitfalls:
- A negated/absence statement does NOT assert the positive. "Dana does not have
  access" does NOT make "has access" present. If the only occurrence of a
  forbidden fact is inside a negation or a past-tense "after X left/died/changed"
  clause that conveys it is no longer true, the forbidden fact is ABSENT.
- An identifier that is a substring of a DIFFERENT, legitimately-surviving longer
  identifier is NOT the forbidden one. Forbidden "TXN-12345" is ABSENT even if a
  surviving row contains "TXN-123456" (a different transaction).
- Surface-form / cross-script variants DO count as present if they refer to the
  same entity (e.g. a Chinese name and its romanization).

A case PASSES iff every MUST-be-present fact is semantically conveyed AND no
MUST-NOT fact is semantically present in the retrieval.

Output EXACTLY one JSON object, nothing else:
{{"pass": true|false, "reason": "<one sentence>"}}
""")


def make_judge():
    import openai
    client = openai.OpenAI(api_key=KEY, base_url=BASE)

    import re
    def judge(rec):
        prompt = JUDGE.format(
            query=rec["final_query"],
            blob=" ".join(rec["retrieved"])[:4000],
            mc=rec["must_contain"] or "(nothing required present)",
            mnc=rec["must_not_contain"] or "(nothing forbidden)")
        last = ""
        for attempt in range(3):
            r = client.chat.completions.create(
                model=MODEL, max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "3000")), temperature=0.0,
                messages=[{"role": "user", "content": prompt}])
            text = r.choices[0].message.content or ""
            last = text
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                d = json.loads(m.group(0))
                return bool(d["pass"]), d.get("reason", "")[:200]
        raise ValueError(f"no json after 3 tries: {last[:80]!r}")
    return judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blobs", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if not KEY:
        sys.exit("set OPENROUTER_API_KEY")

    recs = json.loads(Path(args.blobs).read_text(encoding="utf-8"))
    if args.limit:
        recs = recs[: args.limit]
    slug = Path(args.blobs).stem
    ckpt = OUT / f"nli_{slug}_ckpt.jsonl"
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r
        print(f"resume: {len(done)} judged")

    judge = make_judge()
    fout = ckpt.open("a", encoding="utf-8")
    for i, rec in enumerate(recs):
        if rec["id"] in done:
            continue
        try:
            nli_ok, reason = judge(rec)
        except Exception as e:
            print(f"  [judge err] {rec['id']}: {str(e)[:80]}")
            continue
        out = {"id": rec["id"], "category": rec["category"],
               "substring_pass": rec["substring_pass"], "nli_pass": nli_ok,
               "reason": reason}
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        fout.flush()
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(recs)} judged", flush=True)
    fout.close()

    judged = [json.loads(l) for l in ckpt.read_text(encoding="utf-8").splitlines() if l.strip()]
    judged = [j for j in judged if j["id"] in {r["id"] for r in recs}]
    n = len(judged)
    sub = sum(j["substring_pass"] for j in judged)
    nli = sum(j["nli_pass"] for j in judged)
    flips_up = sum(1 for j in judged if j["nli_pass"] and not j["substring_pass"])
    flips_dn = sum(1 for j in judged if j["substring_pass"] and not j["nli_pass"])
    print(f"\n=== {slug}: {n} cases ===")
    print(f"substring {sub}/{n} = {sub/n:.1%}   NLI {nli}/{n} = {nli/n:.1%}")
    print(f"substring-FAIL but NLI-PASS (scorer false-neg): {flips_up}")
    print(f"substring-PASS but NLI-FAIL (scorer false-pos): {flips_dn}")
    print(f"\n{'category':<26}{'sub':>6}{'nli':>6}{'Δ':>5}")
    bycat = defaultdict(lambda: {"sub": 0, "nli": 0, "n": 0})
    for j in judged:
        d = bycat[j["category"]]
        d["n"] += 1; d["sub"] += j["substring_pass"]; d["nli"] += j["nli_pass"]
    for cat in sorted(bycat):
        d = bycat[cat]
        print(f"{cat:<26}{d['sub']:>3}/{d['n']:<2}{d['nli']:>3}/{d['n']:<2}{d['nli']-d['sub']:>+4}")
    summary = {"system": slug, "n": n, "substring_pass": sub, "nli_pass": nli,
               "scorer_false_neg": flips_up, "scorer_false_pos": flips_dn,
               "by_category": {k: dict(v) for k, v in bycat.items()}}
    (OUT / f"nli_{slug}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote nli_{slug}_summary.json")


if __name__ == "__main__":
    main()

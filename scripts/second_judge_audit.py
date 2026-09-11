"""Cross-judge admission audit using a second non-Qwen judge.

The same admission protocol, re-run with a
second LLM judge from a different model family (DeepSeek-V3) to
measure admission agreement across judges.

We re-judge the 100 IAA-sampled cases (iaa/cases.csv) — these are
already labeled by:
  - Qwen-2.5-72B (original admission judge for v0.5 generation)
  - 10 NLP/CS human annotators (Fleiss kappa 0.958)

Now also by DeepSeek-V3 with the SAME admission prompt.

Output:
    iaa/second_judge_summary.json  agreement matrix
"""
from __future__ import annotations

import csv
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
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LETHE_REPO))
sys.path.insert(0, str(SCRIPTS_DIR))

# Reuse the judge prompt from the generation script
from generate_adversarial_cases import JUDGE_PROMPT, extract_json_object

SF_KEY = os.environ.get("SILICONFLOW_API_KEY") or \
    ""
SF_BASE = "https://api.siliconflow.cn/v1"

# Default: DeepSeek-V3 as second judge (different family from Qwen)
SECOND_JUDGE_MODEL = os.environ.get(
    "SECOND_JUDGE_MODEL", "deepseek-ai/DeepSeek-V3"
)

IAA_DIR = Path(__file__).resolve().parent.parent / "iaa"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = IAA_DIR / "second_judge_summary.json"


def make_judge():
    import openai
    client = openai.OpenAI(api_key=SF_KEY, base_url=SF_BASE)

    def judge(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=SECOND_JUDGE_MODEL,
            max_tokens=512,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""
    return judge


def load_cases_by_id():
    from bench.forgeteval.adversarial import ATTACK_CATEGORIES
    out = {}
    for cat, cases in ATTACK_CATEGORIES.items():
        for c in cases:
            out[c.id] = (cat, c)
    return out


def main():
    cases_by_id = load_cases_by_id()

    # Load the 100 IAA case IDs
    iaa_csv = IAA_DIR / "cases.csv"
    iaa_ids: list[str] = []
    with iaa_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iaa_ids.append(row["case_id"])
    print(f"loaded {len(iaa_ids)} IAA case IDs")

    # Load Qwen judge verdicts for these cases.
    # Two sources:
    # (a) judge_precision_v04.json for hand-crafted core cases
    # (b) adversarial_generated_labels.json for LLM-drafted (all admitted by Qwen)
    qwen_verdicts: dict[str, bool] = {}
    jp_path = DATA_DIR / "judge_precision_v04.json"
    if jp_path.exists():
        jp = json.loads(jp_path.read_text(encoding="utf-8"))
        for r in jp.get("per_case", []):
            if r["well_formed"] is not None:
                qwen_verdicts[r["case_id"]] = r["well_formed"]
    gen_labels_path = LETHE_REPO / "bench" / "forgeteval" / \
        "adversarial_generated_labels.json"
    if gen_labels_path.exists():
        labels = json.loads(gen_labels_path.read_text(encoding="utf-8"))
        # All generated cases were admitted by Qwen judge -> well_formed=True
        for cid in labels:
            qwen_verdicts.setdefault(cid, True)

    # Run second judge on the 100 IAA cases
    judge = make_judge()
    print(f"using second judge: {SECOND_JUDGE_MODEL}\n")
    results = []
    t0 = time.perf_counter()
    for i, cid in enumerate(iaa_ids, 1):
        if cid not in cases_by_id:
            print(f"  [{i}/{len(iaa_ids)}] {cid}: not found, skip")
            continue
        cat, c = cases_by_id[cid]
        case_for_judge = {
            "setup_facts": list(c.setup_facts),
            "mutations": [list(m) for m in c.mutations],
            "final_query": c.final_query,
            "must_contain": list(c.must_contain),
            "must_not_contain": list(c.must_not_contain),
        }
        prompt = JUDGE_PROMPT.format(
            case_json=json.dumps(case_for_judge, ensure_ascii=False, indent=2)
        )
        try:
            raw = judge(prompt)
            v = extract_json_object(raw)
            wf = bool(v.get("well_formed", False))
            reason = str(v.get("reason", ""))
        except Exception as e:
            wf = None
            reason = f"error: {type(e).__name__}: {e}"
        qwen_wf = qwen_verdicts.get(cid)
        agree = (wf == qwen_wf) if (wf is not None and qwen_wf is not None) else None
        results.append({
            "case_id": cid, "category": cat,
            "deepseek_wf": wf, "qwen_wf": qwen_wf,
            "agree": agree,
            "deepseek_reason": reason[:200],
        })
        if i % 10 == 0 or i == len(iaa_ids):
            n_agree = sum(1 for r in results if r["agree"] is True)
            n_disagree = sum(1 for r in results if r["agree"] is False)
            try:
                print(f"  [{i:3}/{len(iaa_ids)}] {cid} -> deepseek={wf}  "
                      f"qwen={qwen_wf}  agree-so-far {n_agree}/{i}",
                      flush=True)
            except UnicodeEncodeError:
                print(f"  [{i:3}/{len(iaa_ids)}] (id) -> deepseek={wf}  "
                      f"qwen={qwen_wf}  agree-so-far {n_agree}/{i}",
                      flush=True)

    wall = time.perf_counter() - t0
    n_total = sum(1 for r in results if r["agree"] is not None)
    n_agree = sum(1 for r in results if r["agree"] is True)
    n_disagree = sum(1 for r in results if r["agree"] is False)

    print(f"\nDONE in {wall:.1f}s")
    print(f"Agreement: {n_agree}/{n_total} = {n_agree/max(n_total,1)*100:.1f}%")
    print(f"Disagreements: {n_disagree}")
    print()
    print("Disagreement breakdown:")
    for r in results:
        if r["agree"] is False:
            try:
                print(f"  {r['case_id']:<35} "
                      f"deepseek={r['deepseek_wf']!s:<5} "
                      f"qwen={r['qwen_wf']!s}")
            except UnicodeEncodeError:
                pass

    summary = {
        "second_judge_model": SECOND_JUDGE_MODEL,
        "n_cases": len(iaa_ids),
        "n_compared": n_total,
        "n_agree": n_agree,
        "n_disagree": n_disagree,
        "agreement_rate": n_agree / max(n_total, 1),
        "wall_seconds": wall,
        "per_case": results,
    }
    OUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()


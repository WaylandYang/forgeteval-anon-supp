"""Sample 30 cases stratified across labels + categories and emit a
CSV for an external annotator to label.  Used for the inter-annotator
agreement (IAA) check in the paper.

Output:
    iaa/cases.csv      annotator-facing CSV (id, category, full case)
    iaa/ground_truth.json    LLM-judge's labels, for offline kappa

Usage:
    py scripts/export_iaa_cases.py [--n 30] [--seed 137]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

LETHE_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LETHE_REPO))

LABELS_FILE = LETHE_REPO / "bench" / "forgeteval" / \
    "adversarial_generated_labels.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "iaa"
OUT_DIR.mkdir(exist_ok=True)


def load_cases_by_id():
    from bench.forgeteval.adversarial import ATTACK_CATEGORIES
    out = {}
    for cat, cases in ATTACK_CATEGORIES.items():
        for c in cases:
            out[c.id] = (cat, c)
    return out


def case_text(c) -> str:
    """Human-readable rendering of a case for the CSV."""
    lines = ["SETUP_FACTS:"]
    for f in c.setup_facts:
        lines.append(f"  - {f}")
    lines.append("MUTATIONS:")
    for m in c.mutations:
        lines.append(f"  - {tuple(m)}")
    lines.append(f"FINAL_QUERY: {c.final_query}")
    lines.append(f"MUST_CONTAIN: {list(c.must_contain)}")
    lines.append(f"MUST_NOT_CONTAIN: {list(c.must_not_contain)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30,
                    help="Cases to sample (default 30).")
    ap.add_argument("--seed", type=int, default=137)
    args = ap.parse_args()

    cases_by_id = load_cases_by_id()
    labels = {}
    if LABELS_FILE.exists():
        try:
            labels = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Stratified sample: ceil(n / num_categories) per category.
    # Ensures every attack category is represented in the IAA sample.
    from collections import defaultdict
    by_cat: dict[str, list[str]] = defaultdict(list)
    for cid, (cat, _) in cases_by_id.items():
        by_cat[cat].append(cid)
    categories = sorted(by_cat.keys())
    per_cat = max(1, args.n // len(categories))
    rng = random.Random(args.seed)
    sample: list[str] = []
    for cat in categories:
        ids = list(by_cat[cat])
        rng.shuffle(ids)
        sample.extend(ids[:per_cat])
    # If we have headroom, fill the remainder with the largest pools.
    remainder = args.n - len(sample)
    if remainder > 0:
        leftover = [cid for cat in categories
                    for cid in by_cat[cat][per_cat:]]
        rng.shuffle(leftover)
        sample.extend(leftover[:remainder])
    rng.shuffle(sample)   # randomize the order annotators see

    # Write annotator-facing CSV.
    csv_path = OUT_DIR / "cases.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "category", "full_case",
                    "your_verdict", "your_reason", "time_seconds"])
        for cid in sample:
            cat, c = cases_by_id[cid]
            w.writerow([cid, cat, case_text(c), "", "", ""])
    print(f"Wrote {len(sample)} cases to {csv_path}")

    # Ground truth: for hand-crafted v0.4 cases, the verdict from
    # judge_precision_v04.json (if available).  For v0.5 generated
    # cases, all are admitted by judge (label != "judge_rejected").
    gt_path = OUT_DIR / "ground_truth.json"
    judge_prec_path = Path(__file__).resolve().parent.parent / \
        "data" / "judge_precision_v04.json"
    judge_results = {}
    if judge_prec_path.exists():
        try:
            jp = json.loads(judge_prec_path.read_text(encoding="utf-8"))
            judge_results = {r["case_id"]: r for r in jp.get("per_case", [])}
        except Exception:
            pass
    gt = {}
    for cid in sample:
        if cid in judge_results:
            gt[cid] = {
                "judge_verdict": judge_results[cid]["well_formed"],
                "judge_reason": judge_results[cid]["reason"],
                "source": "hand_crafted_v0.4 + judge",
            }
        elif cid in labels:
            gt[cid] = {
                "judge_verdict": True,
                "judge_reason": f"admitted by judge with label {labels[cid]}",
                "source": "v0.5_generated",
            }
        else:
            gt[cid] = {
                "judge_verdict": True,
                "judge_reason": "hand_crafted, no judge run",
                "source": "hand_crafted_v0.4",
            }
    gt_path.write_text(json.dumps(gt, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print(f"Wrote ground truth to {gt_path}")


if __name__ == "__main__":
    main()

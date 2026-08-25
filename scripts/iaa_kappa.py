"""Compute Cohen's kappa between human annotator and LLM-judge on
ForgetEval-Adv well-formedness decisions.

Reads:
    iaa/cases.csv          (filled by annotator, column 'your_verdict')
    iaa/ground_truth.json  (judge's verdict per case)

Output: kappa + breakdown + the few disagreements for spot-check.

Usage:
    py scripts/iaa_kappa.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

IAA_DIR = Path(__file__).resolve().parent.parent / "iaa"


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> tuple[float, dict]:
    """Returns (kappa, breakdown). Pairs are (annotator_wf, judge_wf)."""
    if not pairs:
        return 0.0, {}
    n = len(pairs)
    agree = sum(1 for a, b in pairs if a == b)
    p_o = agree / n
    p_a_wf = sum(1 for a, _ in pairs if a) / n
    p_j_wf = sum(1 for _, b in pairs if b) / n
    p_e = p_a_wf * p_j_wf + (1 - p_a_wf) * (1 - p_j_wf)
    if p_e == 1.0:
        return 1.0, {"n": n, "observed": p_o, "expected": p_e}
    kappa = (p_o - p_e) / (1 - p_e)
    return kappa, {
        "n": n,
        "observed_agreement": p_o,
        "expected_agreement": p_e,
        "annotator_wf_rate": p_a_wf,
        "judge_wf_rate": p_j_wf,
        "n_disagreements": n - agree,
    }


def main():
    csv_path = IAA_DIR / "cases.csv"
    gt_path = IAA_DIR / "ground_truth.json"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print("Run scripts/export_iaa_cases.py first, then have an")
        print("annotator fill in the 'your_verdict' column.")
        return
    if not gt_path.exists():
        print(f"ERROR: {gt_path} not found.")
        return

    ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
    pairs: list[tuple[bool, bool]] = []
    rows: list[dict] = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            verdict = row.get("your_verdict", "").strip().lower()
            if verdict not in {"wf", "ill"}:
                continue
            cid = row["case_id"]
            if cid not in ground_truth:
                continue
            ann_wf = verdict == "wf"
            judge_wf = ground_truth[cid]["judge_verdict"]
            pairs.append((ann_wf, judge_wf))
            rows.append({
                "case_id": cid, "category": row["category"],
                "annotator": ann_wf, "judge": judge_wf,
                "annotator_reason": row.get("your_reason", "").strip(),
                "judge_reason": ground_truth[cid]["judge_reason"],
            })

    if not pairs:
        print("No labeled rows found.")
        return

    kappa, breakdown = cohen_kappa(pairs)
    print(f"\nIAA on {len(pairs)} cases")
    print(f"  Observed agreement: {breakdown['observed_agreement']*100:.1f}%")
    print(f"  Expected agreement: {breakdown['expected_agreement']*100:.1f}%")
    print(f"  Cohen's kappa:      {kappa:.3f}")
    print(f"  Disagreements:      {breakdown['n_disagreements']}")
    print()

    # Strength-of-agreement rule of thumb (Landis & Koch 1977)
    if kappa < 0.0:
        strength = "poor / worse than chance"
    elif kappa < 0.2:
        strength = "slight"
    elif kappa < 0.4:
        strength = "fair"
    elif kappa < 0.6:
        strength = "moderate"
    elif kappa < 0.8:
        strength = "substantial"
    else:
        strength = "almost perfect"
    print(f"  Interpretation (Landis & Koch): {strength}")
    print()

    # Show disagreements for paper spot-check.
    disagreements = [r for r in rows if r["annotator"] != r["judge"]]
    if disagreements:
        print(f"=== Disagreements ({len(disagreements)}) ===\n")
        for r in disagreements:
            print(f"  {r['case_id']}  [{r['category']}]")
            print(f"    annotator: {'wf' if r['annotator'] else 'ill'}"
                  f" ({r['annotator_reason'] or '<no reason>'})")
            print(f"    judge:     {'wf' if r['judge'] else 'ill'}"
                  f" ({r['judge_reason'][:80]})")
            print()


if __name__ == "__main__":
    main()

"""Multi-annotator IAA on ForgetEval-Adv admission decisions.

Computes:
  - Fleiss' kappa across all annotators (multi-rater agreement)
  - Per-case agreement distribution (how many annotators agree)
  - Majority-vote consensus
  - Judge vs majority-vote agreement
  - Per-category Fleiss kappa breakdown

Reads:
  iaa/responses/*.csv     one CSV per annotator, columns matching
                          iaa/cases.csv with 'your_verdict' filled
  iaa/ground_truth.json   judge verdicts per case

Usage:
    py scripts/iaa_fleiss.py
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

IAA_DIR = Path(__file__).resolve().parent.parent / "iaa"
RESP_DIR = IAA_DIR / "responses"


def fleiss_kappa(matrix: list[list[int]]) -> tuple[float, dict]:
    """Compute Fleiss' kappa from a matrix where matrix[i][j] is the
    number of annotators who assigned case i to category j.  All rows
    must sum to the same total n (number of annotators per case)."""
    N = len(matrix)            # cases
    if N == 0:
        return 0.0, {"n_cases": 0}
    k = len(matrix[0])         # categories (well-formed / ill-formed)
    n = sum(matrix[0])         # annotators per case
    if n < 2:
        return 1.0, {"n_cases": N, "n_annotators_per_case": n,
                     "note": "trivial agreement (n<2)"}

    # Marginal probability of each category
    totals_per_cat = [sum(row[j] for row in matrix) for j in range(k)]
    grand_total = sum(totals_per_cat)
    p_j = [t / grand_total for t in totals_per_cat]

    # Per-case agreement
    P_i = []
    for row in matrix:
        s = sum(row[j] * (row[j] - 1) for j in range(k))
        P_i.append(s / (n * (n - 1)))
    P_bar = sum(P_i) / N
    P_e = sum(p ** 2 for p in p_j)
    if P_e == 1.0:
        kappa = 1.0 if P_bar == 1.0 else 0.0
    else:
        kappa = (P_bar - P_e) / (1 - P_e)
    return kappa, {
        "n_cases": N,
        "n_annotators_per_case": n,
        "P_bar (observed)": P_bar,
        "P_e (expected)": P_e,
        "category_marginals": p_j,
    }


def interpret_kappa(k: float) -> str:
    """Landis & Koch 1977 strength labels."""
    if k < 0.0:
        return "poor / worse than chance"
    if k < 0.2:
        return "slight"
    if k < 0.4:
        return "fair"
    if k < 0.6:
        return "moderate"
    if k < 0.8:
        return "substantial"
    return "almost perfect"


def load_responses() -> dict[str, dict[str, bool]]:
    """Returns annotator_id -> {case_id: well_formed_bool}."""
    if not RESP_DIR.exists():
        print(f"ERROR: {RESP_DIR} not found.  Annotators should put their")
        print("filled CSVs there as e.g. iaa/responses/alice.csv.")
        return {}
    out: dict[str, dict[str, bool]] = {}
    for f in sorted(RESP_DIR.glob("*.csv")):
        annotator = f.stem
        verdicts: dict[str, bool] = {}
        with f.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                v = (row.get("your_verdict") or "").strip().lower()
                if v == "wf":
                    verdicts[row["case_id"]] = True
                elif v == "ill":
                    verdicts[row["case_id"]] = False
        out[annotator] = verdicts
        print(f"  loaded {annotator}: {len(verdicts)} verdicts")
    return out


def main():
    print(f"Loading from {RESP_DIR}\n")
    responses = load_responses()
    if not responses:
        return
    n_annot = len(responses)
    print(f"\nLoaded {n_annot} annotators.")

    # Find all cases labeled by at least one annotator.
    all_cases = set()
    for verdicts in responses.values():
        all_cases.update(verdicts.keys())
    all_cases = sorted(all_cases)
    print(f"Unique cases: {len(all_cases)}")

    # Build matrix: case x [wf_count, ill_count].
    matrix = []
    case_label_counts: dict[str, Counter] = {}
    for cid in all_cases:
        wf = sum(1 for a, v in responses.items() if v.get(cid) is True)
        ill = sum(1 for a, v in responses.items() if v.get(cid) is False)
        if wf + ill < n_annot:
            print(f"  ! case {cid} only has {wf+ill}/{n_annot} labels")
        matrix.append([wf, ill])
        case_label_counts[cid] = Counter({"wf": wf, "ill": ill})

    kappa, breakdown = fleiss_kappa(matrix)
    print(f"\n=== Fleiss' kappa ===")
    print(f"  N cases:                {breakdown['n_cases']}")
    print(f"  Annotators per case:    {breakdown['n_annotators_per_case']}")
    print(f"  P_bar (observed):       {breakdown['P_bar (observed)']:.3f}")
    print(f"  P_e (expected):         {breakdown['P_e (expected)']:.3f}")
    print(f"  Fleiss' kappa:          {kappa:.3f}")
    print(f"  Interpretation:         {interpret_kappa(kappa)}")

    # Majority vote per case
    print(f"\n=== Per-case agreement ===")
    n_unanimous_wf = sum(1 for c in case_label_counts.values()
                          if c["ill"] == 0)
    n_unanimous_ill = sum(1 for c in case_label_counts.values()
                           if c["wf"] == 0)
    n_majority_wf = sum(1 for c in case_label_counts.values()
                         if c["wf"] > c["ill"] and c["ill"] > 0)
    n_majority_ill = sum(1 for c in case_label_counts.values()
                          if c["ill"] > c["wf"] and c["wf"] > 0)
    n_tied = sum(1 for c in case_label_counts.values()
                  if c["wf"] == c["ill"])
    print(f"  Unanimous well-formed:  {n_unanimous_wf}")
    print(f"  Unanimous ill-formed:   {n_unanimous_ill}")
    print(f"  Majority well-formed:   {n_majority_wf}")
    print(f"  Majority ill-formed:    {n_majority_ill}")
    print(f"  Tied:                   {n_tied}")

    # Judge agreement with majority-vote consensus
    gt_path = IAA_DIR / "ground_truth.json"
    if gt_path.exists():
        try:
            gt = json.loads(gt_path.read_text(encoding="utf-8"))
        except Exception:
            gt = {}
    else:
        gt = {}
    if gt:
        agree, disagree = 0, []
        for cid in all_cases:
            c = case_label_counts[cid]
            majority_wf = c["wf"] > c["ill"]
            judge_wf = gt.get(cid, {}).get("judge_verdict")
            if judge_wf is None:
                continue
            if majority_wf == judge_wf:
                agree += 1
            else:
                disagree.append((cid, c["wf"], c["ill"], judge_wf))
        print(f"\n=== Judge vs human majority ===")
        print(f"  Agreement: {agree}/{len(all_cases)} "
              f"({agree/max(len(all_cases),1)*100:.1f}%)")
        print(f"  Disagreements: {len(disagree)}")
        for cid, wf, ill, judge in disagree:
            j_label = "wf" if judge else "ill"
            print(f"    {cid}  human={wf}wf/{ill}ill, judge={j_label}")

    # Per-category breakdown
    print(f"\n=== Per-category Fleiss kappa ===")
    # Derive category from case_id prefix (adv_<cat>_<num>)
    cases_by_cat: dict[str, list[int]] = defaultdict(list)
    for idx, cid in enumerate(all_cases):
        if cid.startswith("adv_"):
            parts = cid.split("_")
            # adv_substring_trap_01 -> substring_trap
            cat = "_".join(parts[1:-1])
            cases_by_cat[cat].append(idx)
    for cat, idxs in sorted(cases_by_cat.items()):
        if len(idxs) < 2:
            print(f"  {cat:<30} N={len(idxs)} (too few for kappa)")
            continue
        sub_matrix = [matrix[i] for i in idxs]
        k_cat, _ = fleiss_kappa(sub_matrix)
        wf_majority = sum(1 for i in idxs
                          if matrix[i][0] > matrix[i][1])
        print(f"  {cat:<30} N={len(idxs)}  "
              f"kappa={k_cat:.3f} ({interpret_kappa(k_cat)})  "
              f"WF-majority {wf_majority}/{len(idxs)}")

    # Save summary
    summary = {
        "n_annotators": n_annot,
        "n_cases": len(all_cases),
        "fleiss_kappa": kappa,
        "interpretation": interpret_kappa(kappa),
        "P_bar_observed": breakdown["P_bar (observed)"],
        "P_e_expected": breakdown["P_e (expected)"],
        "per_case_label_counts": {
            cid: {"wf": c["wf"], "ill": c["ill"]}
            for cid, c in case_label_counts.items()
        },
    }
    out_path = IAA_DIR / "fleiss_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nWrote summary to {out_path}")


if __name__ == "__main__":
    main()

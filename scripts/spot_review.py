"""Spot-review tool for D + C admission labels.

Samples a small number of cases per label so a human can verify the
judge's well-formedness decisions and the dual-system labels are
correct.  Used to produce the "human-validated subset" claim in the
paper.

Usage:
    py scripts/spot_review.py [--per-label 5] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from textwrap import indent

LETHE_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LETHE_REPO))

LABELS_FILE = LETHE_REPO / "bench" / "forgeteval" / \
    "adversarial_generated_labels.json"


def load_cases_by_id() -> dict:
    """Load all adversarial cases (hand-crafted v0.4 + generated v0.5)
    keyed by case ID."""
    from bench.forgeteval.adversarial import ATTACK_CATEGORIES
    out = {}
    for cat, cases in ATTACK_CATEGORIES.items():
        for c in cases:
            out[c.id] = (cat, c)
    return out


def render_case(case_id: str, cat: str, c, label: str) -> str:
    lines = [
        f"[{label.upper()}]  {case_id}  ({cat})",
        f"  family:      {c.family}",
        f"  setup_facts:",
    ]
    for fact in c.setup_facts:
        lines.append(f"    - {fact}")
    lines.append(f"  mutations:")
    for m in c.mutations:
        lines.append(f"    - {tuple(m)}")
    lines.append(f"  final_query:      {c.final_query!r}")
    lines.append(f"  must_contain:     {list(c.must_contain)}")
    lines.append(f"  must_not_contain: {list(c.must_not_contain)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-label", type=int, default=5,
                    help="Cases to sample per label (default 5).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--labels-only", nargs="*", default=None,
                    help="Only sample these labels.")
    args = ap.parse_args()

    if not LABELS_FILE.exists() or LABELS_FILE.read_text().strip() in ("", "{}"):
        print(f"No labels yet at {LABELS_FILE}.")
        print("Run scripts/generate_adversarial_cases.py first.")
        sys.exit(1)

    labels: dict[str, str] = json.loads(
        LABELS_FILE.read_text(encoding="utf-8")
    )
    cases_by_id = load_cases_by_id()

    by_label: dict[str, list[str]] = {}
    for case_id, label in labels.items():
        by_label.setdefault(label, []).append(case_id)
    # Hand-crafted cases are not in the sidecar — surface them as
    # 'manual' so reviewers can see them too if asked.
    manual_ids = [cid for cid in cases_by_id if cid not in labels]
    if manual_ids:
        by_label["manual"] = manual_ids

    rng = random.Random(args.seed)
    label_order = ["easy", "llm_lift", "llm_regression", "unsolvable",
                   "manual"]
    if args.labels_only:
        label_order = [l for l in label_order if l in args.labels_only]

    print(f"{'='*78}")
    print(f"Spot-review:  {args.per_label} cases per label, seed={args.seed}")
    print(f"{'='*78}\n")
    for label in label_order:
        pool = by_label.get(label, [])
        if not pool:
            print(f"--- {label}: 0 cases in pool, skipping ---\n")
            continue
        sample = rng.sample(pool, min(args.per_label, len(pool)))
        print(f"--- {label} ({len(sample)} of {len(pool)}) ---\n")
        for case_id in sample:
            if case_id not in cases_by_id:
                print(f"  ! {case_id} in labels but missing from bench")
                continue
            cat, c = cases_by_id[case_id]
            print(indent(render_case(case_id, cat, c, label), "  "))
            print()


if __name__ == "__main__":
    main()

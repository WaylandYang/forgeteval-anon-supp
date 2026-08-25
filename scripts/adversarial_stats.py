"""Statistical analysis of adversarial_results.json.

Computes Wilson 95 % confidence intervals per (system, attack category)
and bootstrap CI on the overall score.  Output: a compact comparison
table + bootstrap-based overlap analysis answering "do the deterministic
systems differ significantly on adversarial?"
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "adversarial_results.json"
LABELS_FILE = Path(
    str(Path(__file__).resolve().parent.parent / "bench" / "forgeteval") + "/"
    "adversarial_generated_labels.json"
)


def load_labels() -> dict[str, str]:
    """Load the label sidecar: case_id -> label (one of easy /
    llm_lift / llm_regression / unsolvable).  Hand-crafted v0.4
    cases are not in the sidecar — they get the synthetic label
    'manual' so we can partition the whole suite uniformly."""
    if not LABELS_FILE.exists():
        return {}
    try:
        return json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def case_label(case_id: str, labels: dict[str, str]) -> str:
    return labels.get(case_id, "manual")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95 % confidence interval for a binomial proportion.
    Returns (lower, upper) on [0,1].  Handles k=0 and k=n correctly."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_overall_ci(per_case: list[dict], n_resamples: int = 10000,
                          seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap 95 % CI on the overall pass rate by resampling cases
    with replacement.  Returns (mean, lower, upper).  This treats the
    64 adversarial cases as a sample from a population — the right
    statistical framing if we ask 'how would the score change had we
    drawn a different 64 cases from the same distribution?'"""
    if not per_case:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(per_case)
    flags = [1 if r["passed"] else 0 for r in per_case]
    samples = []
    for _ in range(n_resamples):
        s = sum(flags[rng.randrange(n)] for _ in range(n)) / n
        samples.append(s)
    samples.sort()
    return (sum(flags) / n,
            samples[int(0.025 * n_resamples)],
            samples[int(0.975 * n_resamples)])


def main():
    runs = json.loads(DATA.read_text())

    print(f"{'='*78}")
    print(f"Wilson 95% confidence intervals — adversarial suite, "
          f"64 cases (8 per category)")
    print(f"{'='*78}\n")

    categories = [
        "substring_trap", "prefix_collision", "paraphrase_supersession",
        "negation_trap", "temporal_qualifier", "shared_attribute",
        "compound_fact", "identifier_obfuscation",
        "cross_lingual_identifier", "recursive_supersession",
    ]

    # Per-system × per-category table
    for r in runs:
        print(f"### {r['adapter']}")
        print(f"  {'category':<28}  {'k/n':>5}    {'rate':>6}    95% Wilson CI")
        print(f"  {'-'*28}  {'-'*5}    {'-'*6}    {'-'*16}")
        for cat in categories:
            d = r["by_attack_category"].get(cat, {})
            k, n = d.get("pass", 0), d.get("total", 0)
            lo, hi = wilson_ci(k, n)
            print(f"  {cat:<28}  {k:>2}/{n:<2}    {d.get('rate',0)*100:>5.1f}%    "
                  f"[{lo*100:>4.1f}%, {hi*100:>5.1f}%]")
        # Overall row
        k = r["overall_pass"]; n = r["overall_total"]
        lo, hi = wilson_ci(k, n)
        mean, blo, bhi = bootstrap_overall_ci(r["per_case"])
        print(f"  {'-'*28}  {'-'*5}    {'-'*6}    {'-'*16}")
        print(f"  {'OVERALL (Wilson)':<28}  {k:>2}/{n:<2}    {k/n*100:>5.1f}%    "
              f"[{lo*100:>4.1f}%, {hi*100:>5.1f}%]")
        print(f"  {'OVERALL (bootstrap)':<28}            {mean*100:>5.1f}%    "
              f"[{blo*100:>4.1f}%, {bhi*100:>5.1f}%]")
        print()

    # Pairwise overlap analysis
    print(f"{'='*78}")
    print("Pairwise CI overlap: do these systems differ significantly on")
    print("adversarial overall?  Two systems with non-overlapping Wilson")
    print("intervals can be claimed significantly different at p < 0.05.")
    print(f"{'='*78}\n")

    intervals = []
    for r in runs:
        if r["overall_total"] == 0:
            continue
        lo, hi = wilson_ci(r["overall_pass"], r["overall_total"])
        intervals.append((r["adapter"], r["overall_pass"], r["overall_total"],
                          r["overall_pass"] / r["overall_total"], lo, hi))

    for i, a in enumerate(intervals):
        for b in intervals[i + 1:]:
            overlap = (a[4] <= b[5]) and (b[4] <= a[5])
            sign = "OVERLAP (not sig.)" if overlap else "SEPARATE (p < .05)"
            print(f"  {a[0]:<12} vs {b[0]:<12} | "
                  f"{a[3]*100:>5.1f}% vs {b[3]*100:>5.1f}% | {sign}")

    # Per-category pairwise significance — even when overall CIs
    # overlap, individual categories can have non-overlapping intervals
    # that admit a statistically meaningful claim.
    print()
    print(f"{'='*78}")
    print("Per-category pairwise significance: where do systems separate?")
    print(f"{'='*78}\n")
    for cat in categories:
        cat_intervals = []
        for r in runs:
            d = r["by_attack_category"].get(cat, {})
            k, n = d.get("pass", 0), d.get("total", 0)
            if n == 0:
                continue
            lo, hi = wilson_ci(k, n)
            cat_intervals.append((r["adapter"], k, n, k / n, lo, hi))
        sig_pairs = []
        for i, a in enumerate(cat_intervals):
            for b in cat_intervals[i + 1:]:
                if not ((a[4] <= b[5]) and (b[4] <= a[5])):
                    sig_pairs.append(
                        f"{a[0]} ({a[3]*100:.0f}%) ≠ {b[0]} ({b[3]*100:.0f}%)"
                    )
        if sig_pairs:
            print(f"  {cat}:")
            for p in sig_pairs:
                print(f"    SEPARATE  {p}")

    # Per-label partition: easy / llm_lift / llm_regression / unsolvable
    # plus the hand-crafted v0.4 'manual' bucket.  This is the analytic
    # cut introduced by the D+C admission protocol (§3.3).
    labels = load_labels()
    label_order = ["manual", "easy", "llm_lift", "llm_regression",
                   "unsolvable"]
    print()
    print(f"{'='*78}")
    print("Per-label partition (D + C admission protocol).")
    print(f"  manual          — hand-crafted v0.4 cases (no judge labelling)")
    print(f"  easy            — passes both deterministic Lethe and Lethe+LLM")
    print(f"  llm_lift        — fails Lethe-base, passes Lethe+LLM")
    print(f"  llm_regression  — passes Lethe-base, fails Lethe+LLM (suspect)")
    print(f"  unsolvable      — well-formed per judge, fails on both")
    print(f"{'='*78}\n")

    # Case-count summary (population, not per-system) — same for all
    # adapters since the bench is shared.
    label_counts: dict[str, int] = {lab: 0 for lab in label_order}
    any_run = runs[0] if runs else None
    if any_run:
        for r in any_run["per_case"]:
            lab = case_label(r["id"], labels)
            label_counts[lab] = label_counts.get(lab, 0) + 1
    total_cases = sum(label_counts.values())
    print(f"  Case-count distribution (total = {total_cases}):")
    for lab in label_order:
        c = label_counts.get(lab, 0)
        pct = c / total_cases * 100 if total_cases else 0
        print(f"    {lab:<18}  {c:>4}  ({pct:>4.1f}%)")
    print()

    # Per-system pass rate within each label partition.
    for r in runs:
        print(f"### {r['adapter']}")
        print(f"  {'label':<18}  {'k/n':>8}    {'rate':>6}    95% Wilson CI")
        print(f"  {'-'*18}  {'-'*8}    {'-'*6}    {'-'*16}")
        bucket_pass: dict[str, int] = {lab: 0 for lab in label_order}
        bucket_total: dict[str, int] = {lab: 0 for lab in label_order}
        for pc in r["per_case"]:
            lab = case_label(pc["id"], labels)
            bucket_total[lab] = bucket_total.get(lab, 0) + 1
            if pc["passed"]:
                bucket_pass[lab] = bucket_pass.get(lab, 0) + 1
        for lab in label_order:
            n = bucket_total.get(lab, 0)
            if n == 0:
                continue
            k = bucket_pass.get(lab, 0)
            lo, hi = wilson_ci(k, n)
            print(f"  {lab:<18}  {k:>3}/{n:<4}    "
                  f"{k/n*100:>5.1f}%    "
                  f"[{lo*100:>4.1f}%, {hi*100:>5.1f}%]")
        print()

    # Latency summary
    print()
    print(f"{'='*78}")
    print("Wall-time per case (lower is better).  All adapters run on the")
    print("same machine, same embedder, same cases — directly comparable.")
    print(f"{'='*78}\n")
    print(f"  {'system':<12}  {'wall (s)':>10}  {'per case':>10}  {'rel. to lethe':>15}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*15}")
    lethe_per = None
    for r in runs:
        if r["adapter"] == "lethe":
            lethe_per = r["wall_seconds"] / max(r["case_count"], 1)
            break
    for r in runs:
        n = max(r["case_count"], 1)
        per = r["wall_seconds"] / n
        rel = f"{per/lethe_per:.1f}x" if lethe_per else "—"
        print(f"  {r['adapter']:<12}  {r['wall_seconds']:>10.1f}  "
              f"{per*1000:>8.1f}ms  {rel:>15}")


if __name__ == "__main__":
    main()

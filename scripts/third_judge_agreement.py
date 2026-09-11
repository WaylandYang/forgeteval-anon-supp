"""Compute agreement matrix: Claude (3rd judge) vs Qwen / DeepSeek / Humans.

Outputs: iaa/three_judge_agreement.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

IAA = Path(__file__).resolve().parent.parent / "iaa"

claude = json.loads((IAA / "third_judge_claude_summary.json").read_text(encoding="utf-8"))
qwen_gt = json.loads((IAA / "ground_truth.json").read_text(encoding="utf-8"))
deepseek = json.loads((IAA / "second_judge_summary.json").read_text(encoding="utf-8"))
fleiss = json.loads((IAA / "fleiss_summary.json").read_text(encoding="utf-8"))

claude_v = {c["case_id"]: c["well_formed"] for c in claude["per_case"]}
qwen_v = {cid: rec["judge_verdict"] for cid, rec in qwen_gt.items()}
deepseek_v = {c["case_id"]: c["deepseek_wf"] for c in deepseek["per_case"]}

# Human majority: WF if ≥6/10 say WF
human_v = {}
for cid, counts in fleiss["per_case_label_counts"].items():
    human_v[cid] = counts["wf"] >= 6


def agree(a, b):
    keys = set(a) & set(b)
    n = len(keys)
    matches = sum(1 for k in keys if a[k] == b[k])
    return matches, n, matches / n if n else 0.0


pairs = [
    ("Claude", "Qwen-2.5-72B", claude_v, qwen_v),
    ("Claude", "Human-majority", claude_v, human_v),
    ("Claude", "DeepSeek-V3", claude_v, deepseek_v),
    ("Qwen-2.5-72B", "Human-majority", qwen_v, human_v),
    ("DeepSeek-V3", "Human-majority", deepseek_v, human_v),
    ("Qwen-2.5-72B", "DeepSeek-V3", qwen_v, deepseek_v),
]

print(f"{'Pair':<40} {'agree':>10} {'n':>5} {'rate':>8}")
print("-" * 65)
results = {}
for a_name, b_name, a, b in pairs:
    m, n, r = agree(a, b)
    key = f"{a_name}__vs__{b_name}"
    results[key] = {"matches": m, "total": n, "rate": r}
    print(f"{a_name + ' vs ' + b_name:<40} {m:>10} {n:>5} {r:>8.3f}")

# Distribution of verdicts by judge
print()
print("Per-judge WF/Ill distribution:")
for name, v in [("Qwen", qwen_v), ("Claude", claude_v), ("DeepSeek", deepseek_v), ("Human-maj", human_v)]:
    c = Counter(v.values())
    print(f"  {name:<12} WF={c[True]:>3}  Ill={c[False]:>3}")

# Three-judge unanimous vs split
print()
common = set(claude_v) & set(qwen_v) & set(deepseek_v) & set(human_v)
unanimous_wf = sum(1 for k in common if claude_v[k] and qwen_v[k] and deepseek_v[k] and human_v[k])
unanimous_ill = sum(1 for k in common if not claude_v[k] and not qwen_v[k] and not deepseek_v[k] and not human_v[k])
print(f"Cases unanimous across 3 LLMs + human-majority: WF={unanimous_wf}, Ill={unanimous_ill}")
print(f"At-least-one disagreement: {len(common) - unanimous_wf - unanimous_ill} / {len(common)}")

# Cases where Claude disagrees with Qwen
print()
print("Cases where Claude disagrees with Qwen (showing first 10):")
disagreements = [(k, qwen_v[k], claude_v[k]) for k in set(claude_v) & set(qwen_v) if claude_v[k] != qwen_v[k]]
for cid, qv, cv in disagreements[:10]:
    h = human_v.get(cid, "?")
    d = deepseek_v.get(cid, "?")
    print(f"  {cid:<40} Qwen={qv}  Claude={cv}  DeepSeek={d}  Human-maj={h}")
print(f"Total Claude-Qwen disagreements: {len(disagreements)}")

out = {
    "summary": {
        "n_cases": len(common),
        "agreement_rates": {k: v["rate"] for k, v in results.items()},
        "per_judge_distribution": {
            name: dict(Counter(v.values())) for name, v in [
                ("Qwen-2.5-72B", qwen_v),
                ("Claude", claude_v),
                ("DeepSeek-V3", deepseek_v),
                ("Human-majority", human_v),
            ]
        },
        "unanimous_wf": unanimous_wf,
        "unanimous_ill": unanimous_ill,
        "any_disagreement": len(common) - unanimous_wf - unanimous_ill,
    },
    "pairwise": results,
    "claude_qwen_disagreements": [
        {"case_id": cid, "qwen": qwen_v[cid], "claude": claude_v[cid],
         "deepseek": deepseek_v.get(cid), "human_majority": human_v.get(cid)}
        for cid in set(claude_v) & set(qwen_v) if claude_v[cid] != qwen_v[cid]
    ],
}
(IAA / "three_judge_agreement.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"\nWrote iaa/three_judge_agreement.json")

"""Paired significance tests for the control-plane placement results.

Every ForgetEval-Adv comparison in this paper is a *paired* design: the same
case list is run against every system, so the correct test for "does the
mutation-time hook lift capability?" is McNemar's test on the per-case
verdicts, not a two-sample test on the aggregate rates (which would ignore
the pairing and is what overlapping Wilson intervals silently assume).

Reports, for each system pair:
  * exact McNemar (binomial on the discordant pairs; exact rather than the
    chi-square approximation because several per-category discordant counts
    are below the n>=25 rule of thumb),
  * the paired difference in pass rate with a bootstrap percentile CI,
  * per-category tests with Holm-Bonferroni correction across the 10
    attack categories.

  python scripts/mcnemar_significance.py

Writes data/mcnemar_significance.json and prints a LaTeX table fragment.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest

import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.forgeteval.adversarial import case_to_attack_category

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BOOTSTRAP_N = 10000
SEED = 42


def load_json(name):
    return json.loads((DATA / name).read_text(encoding="utf-8-sig"))


def load_jsonl(name):
    out = []
    for line in (DATA / name).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def verdicts_from_blobs(name, field="substring_pass"):
    return {r["id"]: (r["category"], bool(r[field])) for r in load_json(name)}


def verdicts_from_percase(name, field, id_key="id"):
    rows = load_jsonl(name) if name.endswith(".jsonl") else load_json(name)
    return {r[id_key]: (r.get("category"), bool(r[field])) for r in rows}


def verdicts_from_results(name, adapter=None):
    d = load_json(name)
    entries = d if isinstance(d, list) else [d]
    if adapter:
        entries = [e for e in entries if e.get("adapter") == adapter]
    e = entries[0]
    return {r["id"]: (r["attack_category"], bool(r["passed"]))
            for r in e["per_case"]}


def verdicts_v051(base_name, base_adapter, v051_system):
    """Reconstruct the exact 385-case verdict set behind Table 2.

    Table 2 is the v0.5 365-case run plus the 20 hand-crafted
    identifier_obfuscation cases added in v0.5.1, which were run
    separately. Composing them here means the significance test, the
    per-category deltas, and the table cells all come from one run --
    important because App. M shows the hook's verdict on individual
    compound_fact / identifier_obfuscation cases is not stable across
    re-runs, so a *different* run of the same configuration would give
    the same total with different per-category attribution.
    """
    v = verdicts_from_results(base_name, base_adapter)
    extra = load_json("identifier_obfuscation_v051_results.json")
    for r in extra["by_system"][v051_system]["results"]:
        v[r["case_id"]] = ("identifier_obfuscation", bool(r["pass"]))
    return v


def mcnemar(pairs):
    """pairs: list of (a_pass, b_pass). Returns dict with b, c, p_exact."""
    b = sum(1 for a, x in pairs if a and not x)   # A pass, B fail
    c = sum(1 for a, x in pairs if not a and x)   # A fail, B pass
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "discordant": 0, "p_exact": 1.0}
    p = binomtest(min(b, c), n=n, p=0.5).pvalue
    return {"b": b, "c": c, "discordant": n, "p_exact": min(1.0, p)}


def bootstrap_diff(pairs, n_resamples=BOOTSTRAP_N, seed=SEED):
    """Percentile CI for (rate_B - rate_A), resampling cases (keeps pairing)."""
    rng = random.Random(seed)
    n = len(pairs)
    idx = range(n)
    diffs = []
    for _ in range(n_resamples):
        s = [pairs[rng.choice(idx)] for _ in idx]
        ra = sum(1 for a, _ in s if a) / n
        rb = sum(1 for _, x in s if x) / n
        diffs.append(rb - ra)
    diffs.sort()
    lo = diffs[int(0.025 * n_resamples)]
    hi = diffs[int(0.975 * n_resamples)]
    return lo, hi


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        v = (m - rank) * pvals[i]
        running = max(running, v)
        adj[i] = min(1.0, running)
    return adj


def compare(label, a_name, a, b_name, b):
    common = sorted(set(a) & set(b))
    pairs = [(a[i][1], b[i][1]) for i in common]
    cats = [a[i][0] or b[i][0] for i in common]

    n = len(pairs)
    ra = sum(1 for x, _ in pairs if x)
    rb = sum(1 for _, y in pairs if y)
    overall = mcnemar(pairs)
    lo, hi = bootstrap_diff(pairs)

    by_cat, cat_names, cat_p = {}, [], []
    grouped = defaultdict(list)
    for cat, pr in zip(cats, pairs):
        grouped[cat].append(pr)
    for cat in sorted(grouped):
        g = grouped[cat]
        r = mcnemar(g)
        r["n"] = len(g)
        r["a_pass"] = sum(1 for x, _ in g if x)
        r["b_pass"] = sum(1 for _, y in g if y)
        by_cat[cat] = r
        cat_names.append(cat)
        cat_p.append(r["p_exact"])
    for cat, adj in zip(cat_names, holm(cat_p)):
        by_cat[cat]["p_holm"] = adj

    return {
        "label": label, "n_paired": n,
        "system_a": a_name, "system_b": b_name,
        "a_pass": ra, "b_pass": rb,
        "a_rate": ra / n, "b_rate": rb / n,
        "diff_pt": 100 * (rb - ra) / n,
        "diff_ci95_pt": [100 * lo, 100 * hi],
        "overall": overall,
        "by_category": by_cat,
    }


def fmt_p(p):
    if p < 1e-16:
        return "$<10^{-16}$"
    if p < 0.001:
        return f"${p:.1e}$".replace("e-0", "e-").replace("e-", r"\times 10^{-") + "}$".replace("$$", "$")
    return f"{p:.3f}"


def main():
    results = []

    # --- headline: the configuration Table 2 reports (DeepSeek-V3 hook) ---
    # NB: blobs_lethe_llm.json is the DeepSeek-V4-Pro hook (341/385
    # substring), a different configuration that happens to score 353/385
    # under the NLI scorer -- numerically identical to the V3 substring
    # headline. Reporting the V4-Pro delta beside a V3 table would state
    # two different lifts as if they were one, so the headline test uses
    # the V3 per-case verdicts that Table 2 is actually built from.
    det = verdicts_v051("adversarial_results_v05.json", "lethe", "Lethe")
    hook_v3 = verdicts_v051(
        "adversarial_results_with_llm_siliconflow.json", None, "Lethe+LLM")
    results.append(compare(
        "Lethe deterministic vs Lethe+LLM hook, DeepSeek-V3 (substring, 385)",
        "lethe", det, "lethe+llm(V3)", hook_v3))

    # --- same store, DeepSeek-V4-Pro hook: the blobs we NLI re-scored ---
    hook_v4 = verdicts_from_blobs("blobs_lethe_llm.json")
    results.append(compare(
        "Lethe deterministic vs Lethe+LLM hook, DeepSeek-V4-Pro (substring, 385)",
        "lethe", det, "lethe+llm(V4Pro)", hook_v4))

    # --- same comparison under the NLI scorer (scorer-invariance) ---
    det_nli = verdicts_from_percase("nli_blobs_lethe_percase.jsonl", "nli_pass")
    hook_nli = verdicts_from_percase("nli_blobs_lethe_llm_ckpt.jsonl", "nli_pass")
    results.append(compare(
        "Lethe deterministic vs Lethe+LLM hook (NLI, 385)",
        "lethe", det_nli, "lethe+llm", hook_nli))

    # --- architecture-agnosticism: same hook on LangGraph InMemoryStore ---
    lg = verdicts_v051("adversarial_results_v05.json", "langmem", "LangGraph")
    lg_llm = verdicts_v051(
        "adversarial_results_v05_langgraph_llm.json", None, "LangGraph+LLM")
    results.append(compare(
        "LangGraph deterministic vs LangGraph+LLM hook (substring, 385)",
        "langgraph", lg, "langgraph+llm", lg_llm))

    # --- the headline: repaired suite, probe scoring, full budget ---
    # The hook sides use the re-measured checkpoints. Under the
    # 512-token cap the mutation-time prompts that emit rewritten row
    # text were truncated, so the earlier verdict sets understate the
    # hook; the deterministic sides call no model and are unchanged.
    # Same adapter and same hook on both sides; the only difference is
    # whether the mutation-time LLM is wired in. Scored on the repaired
    # suite with probe-based must_not_contain, so neither side can earn a
    # pass by deleting indiscriminately or by failing to rank a row it
    # never deleted (Sec. "What a non-system scores").
    det07 = verdicts_from_percase(
        "openrouter_hook_deepseek_deepseek-v4-flash_nollm_v07_probed_ckpt.jsonl",
        "ok")
    hook07 = verdicts_from_percase(
        "openrouter_hook_deepseek_deepseek-v4-flash_v07_probed_mt3000repr3_ckpt.jsonl",
        "ok")
    for d in (det07, hook07):
        for k in d:
            d[k] = (case_to_attack_category(k), d[k][1])
    results.append(compare(
        "deterministic vs mutation-time hook (v0.7 suite, probe scoring, 385)",
        "deterministic", det07, "hook", hook07))

    # --- architecture-agnosticism, on the repaired suite this time ---
    # The claim previously rested on the shipped suite alone, which is the
    # version Sec. 5.4 shows was gameable. LangGraph's InMemoryStore shares
    # no code with the reference store; only the hook contract is common.
    lg_det07 = verdicts_from_percase(
        "openrouter_hook_deepseek_deepseek-v4-flash_langgraph_nollm"
        "_v07_probed_ckpt.jsonl", "ok")
    lg_hook07 = verdicts_from_percase(
        "openrouter_hook_deepseek_deepseek-v4-flash_langgraph"
        "_v07_probed_mt3000_ckpt.jsonl", "ok")
    for d in (lg_det07, lg_hook07):
        for k in d:
            d[k] = (case_to_attack_category(k), d[k][1])
    results.append(compare(
        "LangGraph deterministic vs hook (v0.7 suite, probe scoring, 385)",
        "langgraph", lg_det07, "langgraph+hook", lg_hook07))

    # --- null control: the two deterministic backbones should NOT differ ---
    results.append(compare(
        "Lethe vs LangGraph, both deterministic (substring, 365)",
        "lethe", verdicts_from_results("adversarial_results_v05.json", "lethe"),
        "langgraph", lg))

    (DATA / "mcnemar_significance.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    for r in results:
        o = r["overall"]
        print(f"\n=== {r['label']} ===")
        print(f"  n paired          {r['n_paired']}")
        print(f"  {r['system_a']:<16} {r['a_pass']}/{r['n_paired']} = {r['a_rate']:.1%}")
        print(f"  {r['system_b']:<16} {r['b_pass']}/{r['n_paired']} = {r['b_rate']:.1%}")
        print(f"  difference        {r['diff_pt']:+.1f} pt  "
              f"[{r['diff_ci95_pt'][0]:+.1f}, {r['diff_ci95_pt'][1]:+.1f}] boot 95% CI")
        print(f"  discordant b/c    {o['b']}/{o['c']}  (n={o['discordant']})")
        print(f"  McNemar exact p   {o['p_exact']:.3e}")
        sig = [c for c, d in r["by_category"].items() if d["p_holm"] < 0.05]
        print(f"  categories significant after Holm: {len(sig)}/{len(r['by_category'])}"
              + (f"  -> {', '.join(sig)}" if sig else ""))

    print("\nwrote data/mcnemar_significance.json")


if __name__ == "__main__":
    main()

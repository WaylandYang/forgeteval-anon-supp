"""Paper-wide consistency check for headline numbers.

Cross-references claims in abstract, §1 contributions, §5 results,
Appendix tables, §Limitations.  Flags numbers that diverge or are
out of expected range.
"""
from __future__ import annotations
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper" / "paper.tex"
if not PAPER.exists():
    PAPER = ROOT / "paper" / "paper.tex"

def _run_pct(name):
    """Aggregate pass rate of a run, as a percentage, from its result file."""
    import json as _json
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "scripts"))
    from runs import resolve as _resolve
    f = _resolve(name)
    d = _json.loads(f.read_text(encoding="utf-8-sig"))
    return round(100 * d["overall_pass"] / d["overall_total"], 1)


_P = "openrouter_hook_deepseek_deepseek-v4-flash_"

# Truth table -- primary systems read from the runs so this file cannot
# drift from the paper; the rest are stable reference points.
CANON = {
    # Retired: each of these was measured before the survivor and
    # probing requirements, and the appendix that carried it now
    # reports the re-measured figure instead. The evaluable-only
    # aggregates have no successor at all -- every case is evaluated
    # now, so there is no evaluable/strict split left to name.
    "Lethe (det, full)":        _run_pct(_P + "nollm_v07_probed.json"),
    "LangGraph (det, full)":    _run_pct(_P + "langgraph_nollm_v07_probed.json"),
    "MemPalace (full)":         _run_pct(_P + "mempalace_nollm_v07_probed.json"),
    "Mem0 (infer=False, full)": _run_pct(_P + "mem0_nollm_v07_probed.json"),
    "Lethe+LLM (full)":         _run_pct(_P + "v07_probed.json"),
    "LangGraph+LLM (full)":     _run_pct(_P + "langgraph_v07_probed.json"),
    "Mem0+v3 (full)":          _run_pct(_P + "mem0-infer_v07_probed.json"),
    "routed (full)":            _run_pct(_P + "routed_v07_probed.json"),
    "Letta evaluable":        41.0,   # 127/310, N/A excluded
    "Mem0+v3 (full)":          _run_pct(_P + "mem0-infer_v07_probed.json"),
    "Graphiti (full)":          7.5,   # 29/385, all cases evaluated
    # HC subset
    "Lethe HC":               53.0,
    "LangGraph HC":           52.3,
    # 85/132 from tab_hc_split.tex; the 65.2 here predates the
    # re-measurement and was what the prose had wrong.
    "Mem0 HC":                64.4,
    "LangGraph+LLM HC":       94.7,   # 125/132, from tab_hc_split
    "Lethe+LLM HC":             93.2,  # 123/132, from tab_hc_split
    # Cross-LLM (full 385)
    # External subset (77 admitted)
    "Lethe ext":              33.8,
    "LangGraph ext":          32.5,
    # 20/77, re-measured under the v07 requirements (was 22/77)
    "Mem0 ext":               26.0,
    "Lethe+LLM ext":          45.5,
    # 38/77 re-measured (was 39/77 under the pre-repair scoring)
    "LangGraph+LLM ext":      49.4,
    # Graphiti has no current measurement on the external subset --
    # the figure that was here came from the table replaced when
    # three systems were re-run on it, and the paper no longer
    # claims it. Restore only alongside a run.
    # IAA / cohort
    "Fleiss kappa":           0.958,
    # Bench sizes
    "Full 385":               385,
    "HC 132":                 132,
    "LLM-drafted 253":        253,
    "External admitted 77":   77,
    "External raw 80":        80,
    "Annotators":             10,
}



def _count(text, sval):
    """Occurrences of `sval` that are not part of a longer number.

    text.count("98.5") matches inside "98.56", which let a stale
    canonical value report a hit against an unrelated Wilson bound.
    """
    return len(re.findall(r"(?<![\d.])" + re.escape(sval) + r"(?![\d])",
                          text))


def main():
    text = PAPER.read_text(encoding="utf-8")
    print(f"paper.tex: {len(text):,} chars, {text.count(chr(10))+1:,} lines")

    # Find every percentage-like number and its 80-char context.
    pat = re.compile(r"\d+(?:\.\d+)?\s*\\?%")
    matches = list(pat.finditer(text))
    print(f"found {len(matches)} percentage tokens")
    print()

    print("=" * 70)
    print("CANONICAL NUMBERS — check each appears at least once, no near-misses:")
    print("=" * 70)
    flag = 0
    for name, val in CANON.items():
        # Build search patterns for the value
        if isinstance(val, float):
            sval = f"{val:.1f}"
        else:
            sval = str(val)
        hits = _count(text, sval)
        # Also check for close variants (±0.1 for floats)
        if isinstance(val, float):
            near = []
            for off in (-0.2, -0.1, 0.1, 0.2):
                v = round(val + off, 1)
                if v == val:
                    continue
                sv = f"{v:.1f}"
                n = _count(text, sv)
                if n > 0:
                    near.append(f"{sv}({n})")
            near_s = f"  near=[{','.join(near)}]" if near else ""
        else:
            near_s = ""
        mark = "✓" if hits >= 1 else "✗"
        if hits == 0:
            flag += 1
        print(f"  {mark} {name:32s} {sval:>7s}  hits={hits}{near_s}")
    print()
    print(f"missing: {flag}")

    # Spot-check for ANONYMIZED placeholder
    print()
    print("=" * 70)
    print("ANONYMIZED placeholder check:")
    print("=" * 70)
    anon = re.findall(r"https?://[Aa]nonymized|ANONYMIZED|XXX[A-Z]+", text)
    if anon:
        print(f"  ⚠  {len(anon)} ANONYMIZED placeholders still present:")
        for a in set(anon):
            print(f"     '{a}'  ({text.count(a)} occurrences)")
    else:
        print("  ✓ no obvious ANONYMIZED placeholders")

    # Check for hardcoded API keys
    print()
    print("=" * 70)
    print("API key leak check:")
    print("=" * 70)
    keys = re.findall(r"sk-[a-zA-Z0-9]{30,}", text)
    if keys:
        print(f"  ⚠  {len(keys)} potential API keys in paper.tex:")
        for k in set(keys):
            print(f"     {k[:20]}...")
    else:
        print("  ✓ no API keys in paper.tex")


if __name__ == "__main__":
    main()

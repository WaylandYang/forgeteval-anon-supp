"""Apply the v0.6/v0.7 repairs to the external-authored subset.

The 77 externally written cases carry the same defect as the shipped
in-house suite, and for the same reason: whoever writes a
canonicalization case naturally specifies what must disappear and forgets
to specify what must survive. All 8 identifier_obfuscation and all 8
cross_lingual_identifier cases ship with ``must_contain = []``, so a store
that deletes indiscriminately scores 8/8 on both -- which is exactly the
"deterministic 0/8, every LLM-hook 8/8" replication the paper leans on.

That the defect reproduces in independently authored data is itself worth
reporting: it is a property of how people write forgetting tests, not a
slip in one suite.

The repair reuses the in-house machinery unchanged -- mechanically derived
identifier siblings, one authored name pair per script family -- so the
external cases are held to the same standard by the same rules, and the
cross-lingual purge queries are reduced to a single surface form.

  python scripts/repair_external_subset.py            # report
  python scripts/repair_external_subset.py --write    # emit repaired cases
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"

from scripts.repair_canonicalization_cases import (  # noqa: E402
    derive_sibling, TEMPLATE_FACT_A, TEMPLATE_FACT_B, TEMPLATE_QUERY,
)
from scripts.repair_cross_lingual_cases import SIBLINGS  # noqa: E402
from scripts.repair_cross_lingual_queries import reduce_query  # noqa: E402

CANON = ("identifier_obfuscation", "cross_lingual_identifier")

# The external authors wrote codenames and place names in three surface
# forms rather than the structured identifiers the in-house rules cover,
# so two fixed siblings are added with the same three-form shape. One pair
# each, applied uniformly, chosen before any system was re-run -- the same
# discipline as the per-script-family pairs.
CODENAME = ("ORBIT-LINE", "Orbit Line", "orbit_line")
PLACE = ("Cologne", "Köln", "Koln")
CODENAME_RE = __import__("re").compile(r"\b[A-Z][A-Z0-9]+-[A-Z][A-Z0-9]+\b")


def three_form_facts(a, b, c, noun):
    return [f"A separate {noun} is tracked internally as {a}.",
            f"Some records call it {b}.",
            f"The dashboard labels it {c}."]


def script_family(text: str) -> str | None:
    """Which script family a case is written in, by codepoint range."""
    ranges = {
        "Chinese (Han)": (0x4E00, 0x9FFF),
        "Korean (Hangul)": (0xAC00, 0xD7AF),
        "Cyrillic": (0x0400, 0x04FF),
        "Arabic": (0x0600, 0x06FF),
        "Greek": (0x0370, 0x03FF),
        "Hebrew": (0x0590, 0x05FF),
        "Devanagari": (0x0900, 0x097F),
        "Thai": (0x0E00, 0x0E7F),
        "Japanese (Kana)": (0x3040, 0x30FF),
    }
    for ch in text:
        for name, (lo, hi) in ranges.items():
            if lo <= ord(ch) <= hi:
                return name
    return None


def repair(case: dict) -> tuple[dict, str]:
    cat = case.get("category")
    if cat not in CANON or case.get("must_contain"):
        return case, "untouched"

    target = case["mutations"][0][1]
    out = dict(case)

    blob = " ".join(case["setup_facts"]) + " " + target

    if cat == "identifier_obfuscation" and CODENAME_RE.search(blob):
        a, b, c = CODENAME
        out["setup_facts"] = list(case["setup_facts"]) + three_form_facts(
            a, b, c, "deployment project")
        out["final_query"] = f"List every project on file, including {a}."
        out["must_contain"] = [a]
        return out, "codename sibling added"

    if cat == "cross_lingual_identifier" and script_family(blob) is None:
        a, b, c = PLACE
        out["setup_facts"] = list(case["setup_facts"]) + three_form_facts(
            a, b, c, "compliance site")
        out["final_query"] = f"List every site on file, including {a}."
        out["must_contain"] = [a]
        return out, "latin-script sibling added"

    if cat == "identifier_obfuscation":
        d = derive_sibling(target)
        if not d:
            return case, "no rule"
        _, primary, variant = d
        out["setup_facts"] = list(case["setup_facts"]) + [
            TEMPLATE_FACT_A.format(primary=primary),
            TEMPLATE_FACT_B.format(variant=variant)]
        out["final_query"] = TEMPLATE_QUERY.format(primary=primary)
        out["must_contain"] = [primary]
        return out, "sibling added"

    fam = script_family(" ".join(case["setup_facts"]) + target)
    if fam not in SIBLINGS:
        return case, f"no sibling for script {fam!r}"
    native, latin = SIBLINGS[fam]
    out["setup_facts"] = list(case["setup_facts"]) + [
        f"Customer {native} opened a separate account in April.",
        f"The same customer, written {latin}, is on the Premium tier."]
    out["final_query"] = f"List every customer on file, including {latin}."
    out["must_contain"] = [latin]
    out["mutations"] = [
        ([m[0], reduce_query(m[1])] + list(m[2:]))
        if m[0] in ("purge", "release") and any(ord(ch) > 127 for ch in m[1])
        else list(m)
        for m in case["mutations"]]
    return out, "sibling added + query reduced"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    src = json.loads((DATA / "external_subset_cases.json")
                     .read_text(encoding="utf-8-sig"))
    cases = src["admitted_cases"]

    repaired, counts = [], {}
    for c in cases:
        r, how = repair(c)
        repaired.append(r)
        counts[how] = counts.get(how, 0) + 1

    print(f"external cases: {len(cases)}")
    for k, v in sorted(counts.items()):
        print(f"  {k:<34}{v:>4}")
    still_empty = [c["id"] for c in repaired
                   if c["category"] in CANON and not c.get("must_contain")]
    print(f"\ncanonicalization cases still without a positive requirement: "
          f"{len(still_empty)}")
    if still_empty:
        print(f"  {still_empty}")

    if args.write:
        out = dict(src)
        out["admitted_cases"] = repaired
        out["repair"] = ("v0.7: identifier siblings by mechanical rule, "
                         "cross-lingual siblings one pair per script family, "
                         "cross-lingual purge queries reduced to one form")
        (DATA / "external_subset_cases_v07.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\nwrote data/external_subset_cases_v07.json")


if __name__ == "__main__":
    main()

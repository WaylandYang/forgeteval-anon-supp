"""Intake + difficulty-calibration for round-2 external-author cases.

Stage 1 (admission, structural):
  - load every ext2_*.json from --indir
  - normalise the contributor format ({"mutation": {...}}) to the bench format
    ({"mutations": [[...]]})
  - structural well-formedness: valid category, mutation, non-empty final_query,
    no self-trap (must_not_contain is a substring of a must_contain), each
    must_not_contain string occurs in some setup_fact (so it exists pre-mutation)
  -> writes data/external_r2_admitted.json + external_r2_rejected.json

Stage 2 (difficulty calibration, --calibrate):
  - run admitted cases through Lethe (deterministic) and Lethe+LLM
  - label each: 'trivial' (both pass), 'impossible' (both fail),
    'discriminative' (exactly one passes)
  - keep discriminative + a capped quota of trivial/impossible for spread
  -> writes data/external_r2_calibrated.json + a per-category report

  python scripts/intake_external_cases.py --indir external_r2/
  python scripts/intake_external_cases.py --calibrate         # after admission
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data"
CATEGORIES = {
    "substring_trap", "prefix_collision", "paraphrase_supersession",
    "negation_trap", "temporal_qualifier", "shared_attribute",
    "compound_fact", "identifier_obfuscation", "cross_lingual_identifier",
    "recursive_supersession",
}


def normalise(case):
    """Contributor format -> bench format. Returns (norm, error|None)."""
    if not isinstance(case, dict):
        return None, "not an object"
    cid = case.get("id", "")
    cat = case.get("category", "")
    if cat not in CATEGORIES:
        return None, f"bad category {cat!r}"
    setup = case.get("setup_facts", [])
    if not (2 <= len(setup) <= 8):
        return None, "setup_facts must be 2-8"
    m = case.get("mutation")
    if not isinstance(m, dict) or "op" not in m:
        return None, "missing/bad mutation"
    op = m["op"]
    if op == "supersede":
        mut = ["supersede", m.get("old_query", ""), m.get("new_text", "")]
        if not mut[1] or not mut[2]:
            return None, "supersede needs old_query+new_text"
    elif op in ("purge", "release"):
        mut = [op, m.get("target_query", "")]
        if not mut[1]:
            return None, f"{op} needs target_query"
    else:
        return None, f"bad op {op!r}"
    norm = {
        "id": cid, "category": cat, "setup_facts": setup, "mutations": [mut],
        "final_query": case.get("final_query", ""),
        "must_contain": case.get("must_contain", []),
        "must_not_contain": case.get("must_not_contain", []),
        "new_text": mut[2] if op == "supersede" else None,
    }
    return norm, None


def admit(norm):
    """Structural admission. Returns reason or None if admitted."""
    if not norm["final_query"].strip():
        return "empty final_query"
    mc = [s.lower() for s in norm["must_contain"]]
    mnc = [s.lower() for s in norm["must_not_contain"]]
    if not mc and not mnc:
        return "no assertions"
    # self-trap: must_not_contain is a substring of a must_contain
    for t in mnc:
        for s in mc:
            if t in s:
                return f"self-trap: {t!r} substring of must_contain {s!r}"
    # must_not_contain should exist somewhere pre-mutation (setup or new_text)
    haystack = " ".join(norm["setup_facts"]).lower()
    if norm.get("new_text"):
        haystack += " " + norm["new_text"].lower()
    for t in mnc:
        if t not in haystack:
            return f"must_not_contain {t!r} never appears pre-mutation"
    return None


def stage_admission(indir):
    files = sorted(glob.glob(str(ROOT / indir / "ext2_*.json")))
    print(f"found {len(files)} contributor files")
    admitted, rejected = [], []
    seen_ids = set()
    for f in files:
        try:
            cases = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [skip {Path(f).name}] {e}")
            continue
        for c in cases:
            norm, err = normalise(c)
            if err:
                rejected.append({"file": Path(f).name, "case": c.get("id", "?"), "reason": err})
                continue
            if norm["id"] in seen_ids:
                rejected.append({"case": norm["id"], "reason": "duplicate id"})
                continue
            seen_ids.add(norm["id"])
            r = admit(norm)
            if r:
                rejected.append({"case": norm["id"], "reason": r})
            else:
                admitted.append(norm)
    (OUT / "external_r2_admitted.json").write_text(
        json.dumps(admitted, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "external_r2_rejected.json").write_text(
        json.dumps(rejected, ensure_ascii=False, indent=1), encoding="utf-8")
    bycat = defaultdict(int)
    for a in admitted:
        bycat[a["category"]] += 1
    print(f"\nadmitted {len(admitted)}, rejected {len(rejected)}")
    print("per-category admitted:")
    for cat in sorted(CATEGORIES):
        print(f"  {cat:<26} {bycat[cat]}")
    print(f"\nrejection reasons (top):")
    rr = defaultdict(int)
    for r in rejected:
        rr[r["reason"].split(":")[0]] += 1
    for k, v in sorted(rr.items(), key=lambda x: -x[1]):
        print(f"  {v:>3}  {k}")
    print("\nwrote data/external_r2_admitted.json + external_r2_rejected.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="external_r2")
    ap.add_argument("--calibrate", action="store_true",
                    help="(stage 2) pilot admitted cases on systems; needs admitted.json")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.calibrate:
        print("Stage-2 calibration: see calibrate_external.py "
              "(runs Lethe + Lethe+LLM, labels discriminative).")
    else:
        stage_admission(args.indir)

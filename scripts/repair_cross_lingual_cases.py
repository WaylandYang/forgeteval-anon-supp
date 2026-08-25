"""Repair cross_lingual_identifier the same way identifier_obfuscation was
repaired: give every case a second dual-script entity that the purge must
spare, and require it to survive.

Unlike the identifier repair, no rule over the target string can produce
a plausible sibling here -- a sibling for 张伟 has to be a different Han
name, not a string transformation of that one. So the siblings are
authored, but with the smallest discretion we could arrange: **one fixed
pair per script family**, applied to every case in that family, chosen
before any system was re-run. There is no per-case tuning, and the paper
lists this table verbatim so a reader can see exactly what was added and
check that nothing was selected to favour a particular system.

As with the identifier repair, the change is strictly harder for
everyone: any store that passed a case by deleting the whole record now
fails it.

  python scripts/repair_cross_lingual_cases.py --write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"

from bench.forgeteval.adversarial import (  # noqa: E402
    ADVERSARIAL_TESTS, case_to_attack_category,
)

# One sibling entity per script family: (native form, Latin form).
# Each is a common, unambiguous personal name in that script, chosen to
# share no characters or syllables with the targets it sits beside.
SIBLINGS = {
    "Chinese (Han)":   ("李娜", "Li Na"),
    "Korean (Hangul)": ("박지훈", "Park Jihoon"),
    "Cyrillic":        ("Ольга Соколова", "Olga Sokolova"),
    "Arabic":          ("نور الحسن", "Nour Al-Hassan"),
    "Greek":           ("Ελένη Παπαδάκη", "Eleni Papadaki"),
    "Hebrew":          ("דנה לוי", "Dana Levi"),
    "Devanagari":      ("अर्जुन शर्मा", "Arjun Sharma"),
    "Thai":            ("มาลี ศรีสุข", "Malee Srisuk"),
    "Latin only":      ("Renée Dubois", "Renee Dubois"),
}

FACT_NATIVE = "Customer {native} opened a separate account in April."
FACT_LATIN = "The same customer, written {latin}, is on the Premium tier."
QUERY = "List every customer on file, including {latin}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    script_of = {}
    smap = json.loads((DATA / "cross_lingual_script_map.json")
                      .read_text(encoding="utf-8"))
    for fam, ids in smap.items():
        for cid in ids:
            script_of[cid] = fam

    patch, missing, collisions = {}, [], []
    for c in ADVERSARIAL_TESTS:
        if case_to_attack_category(c.id) != "cross_lingual_identifier":
            continue
        fam = script_of.get(c.id)
        if fam not in SIBLINGS:
            missing.append((c.id, fam))
            continue
        native, latin = SIBLINGS[fam]
        # a sibling that collides with this case's own forbidden strings
        # would make the case unpassable; catch it here rather than in a
        # silent score drop
        blob = " ".join(c.setup_facts).lower()
        if native.lower() in blob or latin.lower() in blob:
            collisions.append((c.id, fam, native, latin))
            continue
        patch[c.id] = {
            "category": "cross_lingual_identifier",
            "script_family": fam,
            "add_facts": [FACT_NATIVE.format(native=native),
                          FACT_LATIN.format(latin=latin)],
            "final_query": QUERY.format(latin=latin),
            "must_contain": [latin],
        }

    print(f"patched {len(patch)} / 38 cross_lingual cases")
    from collections import Counter
    print("by script family:",
          Counter(v["script_family"] for v in patch.values()).most_common())
    if missing:
        print(f"\nno script family recorded ({len(missing)}): {missing}")
    if collisions:
        print(f"\nsibling collides with case text ({len(collisions)}):")
        for x in collisions:
            print("   ", x)

    if args.write:
        dest = DATA / "cross_lingual_repair_patch.json"
        dest.write_text(json.dumps(patch, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(f"\nwrote data/{dest.name}")
    else:
        print("\n(dry run; pass --write to emit the patch)")


if __name__ == "__main__":
    main()

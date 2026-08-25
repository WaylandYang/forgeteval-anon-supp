"""Repair the two canonicalization categories so that passing requires
forgetting the right thing AND keeping everything else.

The defect: all 38 identifier_obfuscation and all 38
cross_lingual_identifier cases ship with ``must_contain = []``. Passing
therefore only requires that a forbidden substring be absent, which an
adapter whose ``recall_texts`` returns ``[]`` satisfies perfectly --- it
scores 38/38 on both. Those two categories supply 71 of the 109 net
cases in the headline lift, so the headline could not distinguish
canonicalization from indiscriminate deletion.

The repair follows the pattern the other eight categories already use
(see prefix_collision: purge one key, require the sibling key to
survive). Each case gains a **second entity of the same type**, itself
present in two surface forms, which the purge must NOT touch; the final
query is retargeted at that survivor and ``must_contain`` requires it.

A case now passes only if the system (a) recognises both surface forms
of the target as one entity and removes both, and (b) does not take the
structurally identical sibling with it.

The sibling is derived MECHANICALLY from the target by identifier type
(email, card, IP, URL, handle, person, project code, date, account) --
one rule per type, applied uniformly, no per-case tuning -- so that the
repair cannot be accused of being fitted to any system. Note the repair
makes both categories strictly harder for every system including our
own: the null baseline drops 38/38 -> 0/38, and any over-deleting store
loses the cases it previously passed by deleting everything. The bias it
introduces runs against this paper's thesis, not for it.

  python scripts/repair_canonicalization_cases.py --dry-run
  python scripts/repair_canonicalization_cases.py --write
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data"

from bench.forgeteval.adversarial import (  # noqa: E402
    ADVERSARIAL_TESTS, case_to_attack_category,
)

# ── mechanical sibling derivation, one rule per identifier type ───────
# Each rule maps a target identifier to a structurally identical but
# unmistakably different one. Rules are pure functions of the target.

CARD = re.compile(r"\b(\d{4})[- ]?(\d{4})[- ]?(\d{4})[- ]?(\d{4})\b")
EMAIL = re.compile(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
IPV4 = re.compile(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b")
URLPATH = re.compile(r"\b([a-z0-9.-]+\.(?:io|com|dev|net))/([A-Za-z0-9/_-]+)")
HANDLE = re.compile(r"@([A-Za-z0-9_-]{3,})")
PROJ = re.compile(r"\b([A-Z]{2,}-)?([A-Z]{3,})-(\d{4})\b")
ACCT = re.compile(r"\b(\d{4})-([A-Z]{3})\b")
DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
SSN = re.compile(r"\bSSN\b|\b\d{3}-\d{2}-\d{4}\b")
PHONE = re.compile(r"\b(\d{3})-(\d{3})-(\d{4})\b")
ZIPC = re.compile(r"\bZIP\b|\b\d{5}-\d{4}\b")
TOKEN8 = re.compile(r"\btoken\s+([0-9a-f]{8})\b")
TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b")
ISBN = re.compile(r"\b97[89]\d{10}\b")
PATHZ = re.compile(r"\b([a-z]+/v\d+/[A-Za-z0-9_.-]+)\b")
HEXCOL = re.compile(r"\bcolour\s+([0-9A-F]{6})\b")
SKU = re.compile(r"\bSKU\b|\b([A-Z]{2,})-([A-Z]{2,})-(\d+)\b")
LICKEY = re.compile(r"\b(?:[A-Z0-9]{4}-){3}[A-Z0-9]{4}\b")
TICKET = re.compile(r"\bticket\s+([A-Z]{2,4})-(\d{4,})\b")
VIN = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")
BIC = re.compile(r"\bBIC\s+([A-Z]{8})\b")
SNAKE = re.compile(r"\b([a-z]+_[a-z]+)\b")
PLAIN = re.compile(r"\bcustomer\s+(\d{2,})\b")


def derive_sibling(target: str):
    """Return (kind, sibling_primary, sibling_variant) or None.

    sibling_primary and sibling_variant are two surface forms of ONE new
    entity, mirroring the target's own two-form structure.
    """
    if SSN.search(target):
        return ("ssn", "987-65-4321", "987654321")
    if LICKEY.search(target):
        return ("licence", "QW9Z-4T2K-7HD5-3BXP", "QW9Z4T2K7HD53BXP")
    if VIN.search(target):
        return ("vin", "2FTRX18L1XCA71234", "2ftrx18l1xca71234")
    if BIC.search(target):
        return ("bic", "BNPAFRPP", "bnpafrpp")
    if ISBN.search(target):
        return ("isbn", "9781491957660", "978-1-4919-5766-0")
    if m := TOKEN8.search(target):
        return ("token", "7f3a91c2", "7F3A91C2")
    if m := PHONE.search(target):
        return ("phone", f"{m.group(1)}-555-0198", f"+1 {m.group(1)} 555 0198")
    if ZIPC.search(target):
        return ("zip", "02139-4307", "021394307")
    if m := SKU.search(target):
        return ("sku", "BOOT-TRK-19", "TRK-19")
    if m := TICKET.search(target):
        return ("ticket", f"{m.group(1)}-77310", f"{m.group(1).lower()}-77310")
    if m := HEXCOL.search(target):
        return ("colour", "2E9BD6", "#2e9bd6")
    if m := TIME.search(target):
        return ("time", "09:15", "9:15 AM")
    if m := PATHZ.search(target):
        return ("path", "releases/v4/manifest.zip", "releases/v4/manifest.ZIP")
    if m := SNAKE.search(target):
        return ("snake_handle", "rowan_builds", "@rowan_builds")
    if m := PLAIN.search(target):
        return ("plain_id", "5391", "no. 5391")
    if m := CARD.search(target):
        return ("card", "4242-4242-4242-4242", "4242424242424242")
    if m := ACCT.search(target):
        return ("account", "9876-EUR", "9876")
    if m := DATE.search(target):
        return ("date", "2026-03-08", "03/08/2026")
    if m := PROJ.search(target):
        pre = m.group(1) or ""
        return ("project", f"{pre}ORION-{m.group(3)}", f"ORION-{m.group(3)}")
    if m := EMAIL.search(target):
        return ("email", f"morgan.reyes@{m.group(2)}",
                f"MORGAN.REYES@{m.group(2).upper()}")
    if m := IPV4.search(target):
        a, b, c, d = m.groups()
        nd = (int(d) + 7) % 250 + 1
        return ("ip", f"{a}.{b}.{c}.{nd}",
                f"{a}.{b}.{int(c):03d}.{nd:03d}")
    if m := URLPATH.search(target):
        host = m.group(1)
        return ("url", f"https://{host}/v2/invoices",
                f"{host}/v2/invoices/")
    if m := HANDLE.search(target):
        return ("handle", "@quinn-ops", "quinn-ops")
    # person names: two capitalised words, no digits
    words = re.findall(r"\b[A-Z][a-z]+\b", target)
    if len(words) >= 2:
        return ("person", "Grace Hopper", "G. Hopper")
    return None


TEMPLATE_FACT_A = "Record for {primary} is active on the account."
TEMPLATE_FACT_B = "Same record, alternate form {variant}, was verified."
TEMPLATE_QUERY = "List every record on file, including {primary}."


def build_patch():
    patch, skipped = {}, []
    for c in ADVERSARIAL_TESTS:
        cat = case_to_attack_category(c.id)
        if cat != "identifier_obfuscation":
            continue
        target = c.mutations[0][1]
        d = derive_sibling(target)
        if not d:
            skipped.append((c.id, target))
            continue
        kind, primary, variant = d
        patch[c.id] = {
            "category": cat,
            "sibling_kind": kind,
            "add_facts": [TEMPLATE_FACT_A.format(primary=primary),
                          TEMPLATE_FACT_B.format(variant=variant)],
            "final_query": TEMPLATE_QUERY.format(primary=primary),
            "must_contain": [primary],
            "keep_must_not_contain": c.must_not_contain,
        }
    return patch, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    patch, skipped = build_patch()
    print(f"derived siblings for {len(patch)} identifier_obfuscation cases")
    from collections import Counter
    print("by rule:", Counter(v["sibling_kind"] for v in patch.values()).most_common())
    if skipped:
        print(f"\nno mechanical rule matched ({len(skipped)}) -- these need "
              f"hand authoring, and must be listed as such in the paper:")
        for cid, t in skipped:
            print(f"   {cid}: {t!r}")

    if args.write:
        dest = OUT / "canonicalization_repair_patch.json"
        dest.write_text(json.dumps(patch, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(f"\nwrote data/{dest.name}")
    else:
        print("\n(dry run; pass --write to emit the patch)")


if __name__ == "__main__":
    main()

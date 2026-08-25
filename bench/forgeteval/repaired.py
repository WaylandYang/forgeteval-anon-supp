"""ForgetEval-Adv v0.6: the adversarial suite with the canonicalization
repair applied.

v0.5.1 shipped all 38 identifier_obfuscation and all 38
cross_lingual_identifier cases with ``must_contain = []``. Passing them
required only that a forbidden substring be absent, which an adapter
returning nothing satisfies perfectly. Those 76 cases carried 71 of the
109 net cases in the reported headline lift, so the headline could not
separate canonicalization from indiscriminate deletion.

v0.6 gives each of those cases a second entity of the same kind, present
in two surface forms, that the mutation must spare, and requires it. A
system passes only by collapsing both forms of the target *and* leaving
the structurally identical sibling alone.

Both versions are importable, because every claim that changed needs to
be reportable under each:

    from bench.forgeteval.adversarial import ADVERSARIAL_TESTS  # v0.5.1
    from bench.forgeteval.repaired import REPAIRED_TESTS        # v0.6

Provenance of the added material is deliberately explicit. The
identifier siblings are derived mechanically from the target by
identifier type (24 rules, no per-case tuning); the cross-lingual
siblings are one authored name pair per script family, applied uniformly
to every case in that family. Neither was chosen after seeing a system's
results. See ``scripts/repair_canonicalization_cases.py`` and
``scripts/repair_cross_lingual_cases.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

from .adversarial import ADVERSARIAL_TESTS, case_to_attack_category
from .generate import GeneratedCase

_DATA = Path(__file__).resolve().parent.parent.parent / "data"
_PATCH_FILES = ("canonicalization_repair_patch.json",
                "cross_lingual_repair_patch.json")

REPAIR_PATCH: dict = {}
for _name in _PATCH_FILES:
    _p = _DATA / _name
    if _p.exists():
        REPAIR_PATCH.update(json.loads(_p.read_text(encoding="utf-8")))


def apply_repair(cases=None) -> list:
    """Return the suite with the repair applied to the patched cases."""
    src = cases if cases is not None else ADVERSARIAL_TESTS
    out = []
    for c in src:
        p = REPAIR_PATCH.get(c.id)
        if not p:
            out.append(c)
            continue
        out.append(GeneratedCase(
            id=c.id,
            family=c.family,
            setup_facts=list(c.setup_facts) + list(p["add_facts"]),
            mutations=list(c.mutations),
            final_query=p["final_query"],
            must_contain=list(p["must_contain"]),
            must_not_contain=list(c.must_not_contain),
        ))
    return out


REPAIRED_TESTS: list = apply_repair()

# Cases whose pass criterion changed, so results can be reported both
# ways rather than only under the version that flatters the conclusion.
REPAIRED_IDS = frozenset(REPAIR_PATCH)

__all__ = ["REPAIRED_TESTS", "REPAIRED_IDS", "REPAIR_PATCH",
           "apply_repair", "case_to_attack_category"]

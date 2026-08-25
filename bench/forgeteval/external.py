"""The externally-authored subset, loaded the same way everywhere.

These 77 cases were written by four contributors given the category
schema and nothing else, and they are harder than the ones we write. They
were only ever run against our own two adapters, so the ecosystem columns
of the external table predate the survivor and probing requirements while
the rest of the paper is measured under them -- one table reporting a
system at 8/8 where another reports it at 0/38.

Nothing about the subset made that hard to fix; the loader lived inside
one runner's main() and built a local stand-in for the case dataclass.
It lives here now, returns real GeneratedCase objects, and every runner
can take `--suite external` and be scored by the same rules as the rest.
"""
from __future__ import annotations

import json
import pathlib

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "data"


def load_external_cases():
    """The admitted external cases as GeneratedCase, as the suite uses."""
    from .generate import GeneratedCase

    src = json.loads((DATA / "external_subset_cases_v07.json")
                     .read_text(encoding="utf-8-sig"))
    out = []
    for c in src["admitted_cases"]:
        out.append(GeneratedCase(
            id=c["id"],
            # These carry a category where the suite carries a family; the
            # id encodes the category too, which is what the reporting
            # path reads, so the field is filled for completeness.
            family=c.get("family") or c.get("category", "external"),
            setup_facts=c["setup_facts"],
            mutations=[tuple(m) for m in c["mutations"]],
            final_query=c["final_query"],
            must_contain=c.get("must_contain", []),
            must_not_contain=c.get("must_not_contain", []),
        ))
    return out


def external_category(case_id: str) -> str:
    """Category for an external case id.

    The external ids do not follow the adv_<category>_NN convention that
    case_to_attack_category parses, so reporting them through it labels
    every row "unknown". The mapping is built from the subset file, which
    carries the category the contributor was writing to.
    """
    global _CATMAP
    if _CATMAP is None:
        src = json.loads((DATA / "external_subset_cases_v07.json")
                         .read_text(encoding="utf-8-sig"))
        _CATMAP = {c["id"]: c["category"] for c in src["admitted_cases"]}
    return _CATMAP.get(case_id, "unknown")


_CATMAP = None

__all__ = ["load_external_cases", "external_category"]

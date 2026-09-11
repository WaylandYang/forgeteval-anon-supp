"""Close the second cross_lingual_identifier defect: the purge query names
both surface forms.

v0.6 gave every cross_lingual case a sibling that must survive, which shut
the null/nuke hole. It left a separate leak untouched: all 38 purge queries
spell out *both* forms of the target --

    purge("customer Zhang Wei aka 张伟")

so a store can delete both rows by literal matching and never demonstrate
that it knows 张伟 and Zhang Wei denote one person. That is the whole
capability the category claims to measure, and the query hands it over.

The repair drops every whitespace token containing a non-ASCII letter from
the purge query, leaving the Latin form alone:

    purge("customer Zhang Wei")

It is one rule applied to all 38 cases, with no per-case authoring and no
inspection of any system's output, so it cannot have been fitted to a
result. The store still *contains* the correspondence (the setup facts say
"the same customer in pinyin: ..."), so the case remains satisfiable by a
system that reads its own memory -- what is removed is only the shortcut of
reading the answer off the request.

  python scripts/repair_cross_lingual_queries.py            # validate
  python scripts/repair_cross_lingual_queries.py --write    # emit patch

Gate: null / nuke / a Unicode-normalising store must all collapse on the
category, while an oracle that deletes exactly the forbidden facts must
still pass it. A repair that fails either half is not a repair.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"

from bench.forgeteval.adversarial import case_to_attack_category  # noqa: E402
from bench.forgeteval.repaired import REPAIRED_TESTS  # noqa: E402
from scripts.run_degenerate_baselines import (  # noqa: E402
    NullAdapter, NukeAdapter, NormalizingStore,
)

CONNECTORS = {"aka", "a.k.a.", "/", "-", "and", "or", "vs", "vs.", "|"}


def has_nonascii(s: str) -> bool:
    return any(ord(ch) > 127 for ch in s)


def reduce_query(q: str) -> str:
    """Drop tokens carrying a non-ASCII form, then trim dangling connectors."""
    kept = [t for t in q.split() if not has_nonascii(t)]
    while kept and kept[-1].strip(".,").casefold() in CONNECTORS:
        kept.pop()
    while kept and kept[0].strip(".,").casefold() in CONNECTORS:
        kept.pop(0)
    return " ".join(kept).strip(" /-|")


def patch_case(case):
    muts, changed = [], False
    for m in case.mutations:
        if m[0] in ("purge", "release") and has_nonascii(m[1]):
            reduced = reduce_query(m[1])
            if reduced and any(ch.isalpha() for ch in reduced):
                muts.append((m[0], reduced) + tuple(m[2:]))
                changed = True
                continue
        muts.append(m)
    return (replace(case, mutations=muts), changed) if changed else (case, False)


def build_suite():
    out, n = [], 0
    for c in REPAIRED_TESTS:
        if case_to_attack_category(c.id) == "cross_lingual_identifier":
            c, ch = patch_case(c)
            n += ch
        out.append(c)
    return out, n


class OracleAdapter:
    """Deletes exactly the rows a correct system should delete: any row
    containing a forbidden substring. Establishes the case is satisfiable."""

    name = "oracle"

    def __init__(self, forbidden):
        self.forbidden = forbidden
        self.rows = []

    def reset(self):
        self.rows = []

    def inscribe(self, text):
        self.rows.append(text)
        return ""

    def recall_texts(self, query, k=5):
        return self.rows[:k]

    def _delete(self, query):
        before = len(self.rows)
        self.rows = [r for r in self.rows
                     if not any(f in r for f in self.forbidden)]
        return before - len(self.rows)

    def supersede(self, old_query, new_text):
        self._delete(old_query)
        self.rows.append(new_text)

    def release(self, query):
        return self._delete(query)

    def purge(self, query):
        return self._delete(query)


def evaluate(make_adapter, suite):
    by_cat = defaultdict(lambda: {"pass": 0, "total": 0})
    passed = 0
    for c in suite:
        ok = c.run(make_adapter(c))
        cat = case_to_attack_category(c.id)
        by_cat[cat]["total"] += 1
        if ok:
            by_cat[cat]["pass"] += 1
            passed += 1
    return passed, dict(by_cat)


CAT = "cross_lingual_identifier"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="emit data/cross_lingual_query_patch.json")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    before = list(REPAIRED_TESTS)
    after, n_changed = build_suite()
    print(f"queries reduced to a single surface form: {n_changed}/38\n")

    baselines = [
        ("null", lambda c: NullAdapter()),
        ("nuke", lambda c: NukeAdapter()),
        ("normalize", lambda c: NormalizingStore()),
        ("oracle", lambda c: OracleAdapter(c.must_not_contain)),
    ]

    print(f"{'baseline':<12}{'v0.6 ' + CAT:>34}{'v0.7 (query reduced)':>26}")
    rows = {}
    for name, mk in baselines:
        _, b = evaluate(mk, before)
        pa, a = evaluate(mk, after)
        bc, ac = b[CAT], a[CAT]
        rows[name] = {"v06": bc, "v07": ac, "v07_overall": pa}
        print(f"{name:<12}{bc['pass']:>20}/{bc['total']:<3}"
              f"{ac['pass']:>21}/{ac['total']:<3}"
              f"   overall {pa}/{len(after)}")

    ok = (rows["null"]["v07"]["pass"] == 0
          and rows["nuke"]["v07"]["pass"] == 0
          and rows["normalize"]["v07"]["pass"] <= 4
          and rows["oracle"]["v07"]["pass"] >= 34)
    print(f"\ngate: null=0, nuke=0, normalize<=4, oracle>=34  ->  "
          f"{'PASS' if ok else 'FAIL'}")

    if args.write:
        patch = {}
        for c in after:
            if case_to_attack_category(c.id) == CAT:
                patch[c.id] = {"mutations": [list(m) for m in c.mutations]}
        (DATA / "cross_lingual_query_patch.json").write_text(
            json.dumps(patch, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote data/cross_lingual_query_patch.json ({len(patch)} cases)")

    (DATA / "cross_lingual_query_validation.json").write_text(
        json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

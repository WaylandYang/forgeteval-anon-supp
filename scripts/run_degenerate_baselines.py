"""Degenerate lower bounds for ForgetEval-Adv.

A forgetting benchmark scores a system for *not* returning something. That
makes it uniquely vulnerable to a failure mode recall benchmarks cannot
have: a store that returns nothing, or deletes everything, looks perfect.
Any such benchmark must therefore publish what its own metric gives to
adapters that are not memory systems at all. We did not do this in the
first version of this work, and it cost us: 76 of our 385 adversarial
cases turned out to carry no positive requirement, so the two categories
that carried our headline finding could be passed by returning nothing.

Three controls, each isolating a different way to score well without
forgetting correctly:

  null       recall_texts() -> [] always. Nothing is ever retrieved, so
             no forbidden substring can appear. Pure metric exploit.
  nuke       stores facts, but wipes the whole store on any mutation.
             Over-deletion taken to its limit.
  normalize  a plain lexical store whose only addition is Unicode
             hygiene on the match path (NFKC, casefold, combining-mark
             strip, separator strip, email plus-tag strip). Tests how
             much of the "canonicalization" gap needs an LLM at all.

  python scripts/run_degenerate_baselines.py
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data"

from bench.forgeteval.adversarial import (  # noqa: E402
    ADVERSARIAL_TESTS, case_to_attack_category,
)


def normalize(s: str) -> str:
    """Five lines of Unicode hygiene. No model, no embedding, no lookup."""
    s = unicodedata.normalize("NFKC", s).casefold()
    s = "".join(c for c in unicodedata.normalize("NFKD", s)
                if not unicodedata.combining(c))
    s = re.sub(r"\+[^@\s]*(?=@)", "", s)        # email plus-tag
    s = re.sub(r"[\s\-_./()]+", "", s)          # separators
    return s


class NullAdapter:
    """Retrieves nothing, ever. Not a memory system: the floor that any
    must_not_contain-only case is unable to distinguish from a correct one."""
    name = "null"

    def reset(self): pass
    def inscribe(self, text): return ""
    def recall_texts(self, query, k=5): return []
    def supersede(self, old_query, new_text): pass
    def release(self, query): return 0
    def purge(self, query): return 0


class NukeAdapter:
    """Stores facts and retrieves them, but any mutation wipes everything.
    Maximal collateral damage; scores well wherever nothing is required
    to survive."""
    name = "nuke"

    def reset(self): self.rows = []
    def inscribe(self, text): self.rows.append(text); return ""
    def recall_texts(self, query, k=5): return self.rows[:k]
    def supersede(self, old_query, new_text): self.rows = [new_text]
    def release(self, query): self.rows = []; return 0
    def purge(self, query): self.rows = []; return 0


class NormalizingStore:
    """Plain lexical store; the only thing it adds over a naive one is that
    matching happens on the normalized form. This is the baseline that
    decides whether 'canonicalization' is evidence about LLM placement or
    about Unicode handling."""
    name = "normalize"

    def reset(self): self.rows = []
    def inscribe(self, text): self.rows.append(text); return ""

    def _score(self, q):
        nq = normalize(q)
        toks = [t for t in re.split(r"\s+", q.casefold()) if len(t) > 2]
        out = []
        for i, r in enumerate(self.rows):
            nr = normalize(r)
            s = sum(1 for t in toks if normalize(t) and normalize(t) in nr)
            if nq and nq in nr:
                s += 5
            out.append((s, i))
        return sorted(out, reverse=True)

    def recall_texts(self, query, k=5):
        return [self.rows[i] for s, i in self._score(query)[:k] if s > 0]

    def supersede(self, old_query, new_text):
        sc = self._score(old_query)
        if sc and sc[0][0] > 0:
            self.rows.pop(sc[0][1])
        self.rows.append(new_text)

    def _delete(self, query):
        sc = [(s, i) for s, i in self._score(query) if s > 0]
        if not sc:
            return 0
        top = sc[0][0]
        for i in sorted([i for s, i in sc if s == top], reverse=True):
            self.rows.pop(i)
        return sum(1 for s, _ in sc if s == top)

    def release(self, query): return self._delete(query)
    def purge(self, query): return self._delete(query)


def evaluate(adapter):
    by_cat = defaultdict(lambda: {"pass": 0, "total": 0})
    passed = 0
    for c in ADVERSARIAL_TESTS:
        ok = c.run(adapter)
        cat = case_to_attack_category(c.id)
        by_cat[cat]["total"] += 1
        if ok:
            by_cat[cat]["pass"] += 1
            passed += 1
    return passed, dict(by_cat)


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    # how many cases have no positive requirement at all
    empty = defaultdict(lambda: {"empty": 0, "total": 0})
    for c in ADVERSARIAL_TESTS:
        cat = case_to_attack_category(c.id)
        empty[cat]["total"] += 1
        if not c.must_contain:
            empty[cat]["empty"] += 1

    print("Cases with must_contain = [] (nothing is required to survive):")
    for cat in sorted(empty, key=lambda k: -empty[k]["empty"]):
        d = empty[cat]
        if d["empty"]:
            print(f"  {cat:<28}{d['empty']:>3}/{d['total']:<4} "
                  f"{d['empty']/d['total']:.0%}")
    tot_empty = sum(d["empty"] for d in empty.values())
    print(f"  {'TOTAL':<28}{tot_empty:>3}/385   {tot_empty/385:.0%}\n")

    results = {}
    for cls in (NullAdapter, NukeAdapter, NormalizingStore):
        a = cls()
        passed, by_cat = evaluate(a)
        results[a.name] = {"overall_pass": passed, "overall_total": 385,
                           "overall_rate": passed / 385, "by_category": by_cat}
        print(f"=== {a.name} === {passed}/385 = {passed/385:.1%}")
        for cat in sorted(by_cat):
            d = by_cat[cat]
            mark = "   <-- full marks without forgetting" \
                if d["pass"] == d["total"] else ""
            print(f"  {cat:<28}{d['pass']:>3}/{d['total']:<4}"
                  f"{d['pass']/d['total']:>6.0%}{mark}")
        print()

    results["cases_without_positive_requirement"] = {
        k: v for k, v in empty.items() if v["empty"]}
    (OUT / "degenerate_baselines.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print("wrote data/degenerate_baselines.json")


if __name__ == "__main__":
    main()

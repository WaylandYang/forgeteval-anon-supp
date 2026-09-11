"""Apply the canonicalization repair in memory and check it does what it
is supposed to do, before any of it is committed to the suite.

Three checks, in order of what would embarrass us most:

  1. No self-traps. The required survivor must not contain any forbidden
     substring, and no forbidden substring may appear in the added facts.
     A self-trapping case is unpassable and would silently depress every
     system's score.
  2. The degenerate baselines must lose the category. null went 38/38
     under the old cases; if it still scores above 0 the repair has not
     closed the hole.
  3. A correct reference behaviour must still pass. We model it directly
     (delete exactly the two target forms, keep the rest) so that the
     repair is not merely hard but actually satisfiable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"

from bench.forgeteval.adversarial import (  # noqa: E402
    ADVERSARIAL_TESTS, case_to_attack_category,
)
from bench.forgeteval.generate import GeneratedCase  # noqa: E402


def repaired_cases(patch):
    """Yield the repaired version of every patched case."""
    out = []
    for c in ADVERSARIAL_TESTS:
        p = patch.get(c.id)
        if not p:
            out.append(c)
            continue
        out.append(GeneratedCase(
            id=c.id, family=c.family,
            setup_facts=list(c.setup_facts) + p["add_facts"],
            mutations=list(c.mutations),
            final_query=p["final_query"],
            must_contain=p["must_contain"],
            must_not_contain=c.must_not_contain,
        ))
    return out


def check_self_traps(cases, patch):
    bad = []
    for c in cases:
        if c.id not in patch:
            continue
        for mc in c.must_contain:
            for mnc in c.must_not_contain:
                if mnc.lower() in mc.lower():
                    bad.append((c.id, "survivor contains forbidden", mc, mnc))
        for f in patch[c.id]["add_facts"]:
            for mnc in c.must_not_contain:
                if mnc.lower() in f.lower():
                    bad.append((c.id, "added fact contains forbidden", f, mnc))
        # the survivor must actually be present in some setup fact
        if not any(mc.lower() in f.lower()
                   for mc in c.must_contain for f in c.setup_facts):
            bad.append((c.id, "survivor absent from setup facts",
                        c.must_contain, ""))
    return bad


class Null:
    name = "null"
    def reset(self): pass
    def inscribe(self, t): return ""
    def recall_texts(self, q, k=5): return []
    def supersede(self, oq, nt): pass
    def release(self, q): return 0
    def purge(self, q): return 0


class Nuke:
    name = "nuke"
    def reset(self): self.rows = []
    def inscribe(self, t): self.rows.append(t); return ""
    def recall_texts(self, q, k=5): return self.rows[:k]
    def supersede(self, oq, nt): self.rows = [nt]
    def release(self, q): self.rows = []; return 0
    def purge(self, q): self.rows = []; return 0


class Oracle:
    """Deletes exactly the facts that contain a forbidden substring, and
    nothing else. Not a real system -- the upper bound that proves a
    repaired case is satisfiable at all."""
    name = "oracle"
    def __init__(self, case): self.case = case
    def reset(self): self.rows = []
    def inscribe(self, t): self.rows.append(t); return ""
    def recall_texts(self, q, k=5): return self.rows[:k]
    def _kill(self):
        self.rows = [r for r in self.rows
                     if not any(m.lower() in r.lower()
                                for m in self.case.must_not_contain)]
    def supersede(self, oq, nt): self._kill(); self.rows.append(nt)
    def release(self, q): self._kill(); return 0
    def purge(self, q): self._kill(); return 0


def score(cases, make_adapter, only=None):
    from collections import defaultdict
    by = defaultdict(lambda: {"p": 0, "n": 0})
    for c in cases:
        cat = case_to_attack_category(c.id)
        if only and cat not in only:
            continue
        a = make_adapter(c)
        by[cat]["n"] += 1
        if c.run(a):
            by[cat]["p"] += 1
    return dict(by)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    patch = json.loads((DATA / "canonicalization_repair_patch.json")
                       .read_text(encoding="utf-8"))
    patch.update(json.loads((DATA / "cross_lingual_repair_patch.json")
                            .read_text(encoding="utf-8")))
    cases = repaired_cases(patch)
    cats = {"identifier_obfuscation", "cross_lingual_identifier"}
    print(f"patched cases: {len(patch)}\n")

    print("[1] self-trap check")
    bad = check_self_traps(cases, patch)
    if bad:
        print(f"    FAIL: {len(bad)} problem(s)")
        for b in bad[:10]:
            print("      ", b)
        return
    print("    OK: no self-traps, every survivor present in setup facts\n")

    print("[2] degenerate baselines on the repaired category")
    for cls in (Null, Nuke):
        before = {"null": "38/38", "nuke": "38/38"}[cls.name]
        r = score(cases, lambda c, C=cls: C(), only=cats)
        for cat in sorted(cats):
            d = r[cat]
            verdict = "CLOSED" if d["p"] == 0 else "STILL OPEN"
            print(f"    {cls.name:<6} {cat:<26} was 38/38 -> now "
                  f"{d['p']}/{d['n']}   [{verdict}]")

    print("\n[3] satisfiability: an oracle that deletes exactly the "
          "forbidden facts")
    r = score(cases, lambda c: Oracle(c), only=cats)
    for cat in sorted(cats):
        d = r[cat]
        print(f"    oracle {cat:<26}{d['p']}/{d['n']}"
              + ("   OK" if d["p"] == d["n"] else "   FAIL: unsatisfiable"))


if __name__ == "__main__":
    main()

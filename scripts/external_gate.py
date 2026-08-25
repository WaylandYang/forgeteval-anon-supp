"""Run the reference points on the external subset.

The paper states that the requirements pass the same gate on the
externally-authored cases -- an oracle still passes, null and nuke go to
zero on the canonicalization categories -- and quotes three figures for
it. Those figures had no current run behind them, and the gate is exactly
the claim a reviewer doubting the suite would want checked.

None of these adapters calls a model, so this is seconds of CPU.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bench.forgeteval.external import load_external_cases, external_category  # noqa: E402
from bench.forgeteval.scoring import run_scored  # noqa: E402
from run_degenerate_baselines import (  # noqa: E402
    NullAdapter, NukeAdapter, NormalizingStore,
)

CANON = ("identifier_obfuscation", "cross_lingual_identifier")


class OracleAdapter:
    """Deletes exactly the rows carrying a forbidden string.

    The satisfiability control: not a ceiling, since it removes whole
    rows and so fails every case needing a partial edit.
    """

    name = "oracle"

    def __init__(self):
        self.rows: list[str] = []
        self._forbidden: list[str] = []

    def arm(self, forbidden):
        self._forbidden = [f.lower() for f in forbidden]

    def reset(self):
        self.rows = []

    def inscribe(self, text):
        self.rows.append(text)
        return len(self.rows) - 1

    def recall_texts(self, query, k=5):
        return [r for r in self.rows
                if not any(f in r.lower() for f in self._forbidden)][:k]

    def supersede(self, old_query, new_text):
        self.inscribe(new_text)

    def release(self, query):
        return 0

    def purge(self, query):
        return 0


def evaluate(adapter, cases, arm=False):
    by = {}
    passed = 0
    for c in cases:
        adapter.reset()
        if arm:
            adapter.arm(c.must_not_contain)
        ok = run_scored(c, adapter, probed=True)
        cat = external_category(c.id)
        d = by.setdefault(cat, {"pass": 0, "total": 0})
        d["total"] += 1
        if ok:
            d["pass"] += 1
            passed += 1
    return passed, by


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cases = load_external_cases()
    out = {}
    for cls, arm in ((NullAdapter, False), (NukeAdapter, False),
                     (NormalizingStore, False), (OracleAdapter, True)):
        a = cls()
        p, by = evaluate(a, cases, arm=arm)
        canon_p = sum(by.get(c, {"pass": 0})["pass"] for c in CANON)
        canon_n = sum(by.get(c, {"total": 0})["total"] for c in CANON)
        out[a.name] = {"overall_pass": p, "overall_total": len(cases),
                       "by_category": by,
                       "canonicalization_pass": canon_p,
                       "canonicalization_total": canon_n}
        print("%-11s %2d/%d   canonicalization %d/%d"
              % (a.name, p, len(cases), canon_p, canon_n))
    (ROOT / "data" / "external_gate.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote data/external_gate.json")


if __name__ == "__main__":
    main()

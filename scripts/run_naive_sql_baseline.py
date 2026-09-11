"""A naive SQL baseline on the full 385-case ForgetEval-Adv.

A pure-SQLite + FTS5 store with NO vector recall, NO LLM, and the most
standard naive mutation semantics a backend engineer would reach for:

  - recall_texts : FTS5 BM25 top-k (lexical only)
  - inscribe     : INSERT into the FTS5 table
  - supersede(q, new) : delete the single best BM25 match of q, insert new
  - release(q)   : delete BM25 matches of q (lexical hard-delete)
  - purge(q)     : DELETE WHERE text LIKE '%q%'  (substring hard-delete)

This is exactly the "DELETE FROM memories WHERE ... + FTS5/BM25 reindex"
baseline to compare against.  We score with the same
deterministic substring scorer (GeneratedCase.run) as every other system,
and dump per-case verdicts + the retrieved blob to data/ for the
substring-scorer audit (R2 2.5).

Run:  python scripts/run_naive_sql_baseline.py
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

LETHE_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LETHE_REPO))
OUT = Path(__file__).resolve().parent.parent / "data"

from bench.forgeteval.adversarial import (  # noqa: E402
    ADVERSARIAL_TESTS, case_to_attack_category,
)

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _fts_query(text: str) -> str:
    """Build a safe FTS5 OR-query from the content words of `text`."""
    toks = _TOKEN.findall(text)
    if not toks:
        return None
    # quote each token to neutralise FTS5 operators; OR them together
    return " OR ".join(f'"{t}"' for t in toks)


class NaiveSQLAdapter:
    name = "naive-sql"

    def __init__(self):
        self.db = None

    def reset(self) -> None:
        if self.db is not None:
            self.db.close()
        self.db = sqlite3.connect(":memory:")
        self.db.execute(
            "CREATE VIRTUAL TABLE mem USING fts5(text, tokenize='unicode61')"
        )

    def inscribe(self, text: str) -> int:
        cur = self.db.execute("INSERT INTO mem(text) VALUES (?)", (text,))
        return cur.lastrowid

    def recall_texts(self, query: str, k: int = 5) -> list[str]:
        q = _fts_query(query)
        if q:
            try:
                rows = self.db.execute(
                    "SELECT text FROM mem WHERE mem MATCH ? "
                    "ORDER BY bm25(mem) LIMIT ?",
                    (q, k),
                ).fetchall()
                if rows:
                    return [r[0] for r in rows]
            except sqlite3.OperationalError:
                pass
        # fallback: most-recent rows
        rows = self.db.execute(
            "SELECT text FROM mem ORDER BY rowid DESC LIMIT ?", (k,)
        ).fetchall()
        return [r[0] for r in rows]

    def _best_match_rowid(self, query: str):
        q = _fts_query(query)
        if not q:
            return None
        try:
            row = self.db.execute(
                "SELECT rowid FROM mem WHERE mem MATCH ? "
                "ORDER BY bm25(mem) LIMIT 1",
                (q,),
            ).fetchone()
            return row[0] if row else None
        except sqlite3.OperationalError:
            return None

    def supersede(self, old_query: str, new_text: str) -> None:
        rid = self._best_match_rowid(old_query)
        if rid is not None:
            self.db.execute("DELETE FROM mem WHERE rowid = ?", (rid,))
        self.inscribe(new_text)

    def release(self, query: str) -> int:
        q = _fts_query(query)
        if not q:
            return 0
        try:
            cur = self.db.execute("DELETE FROM mem WHERE mem MATCH ?", (q,))
            return cur.rowcount
        except sqlite3.OperationalError:
            return 0

    def purge(self, query: str) -> int:
        # the canonical naive hard-delete: substring match on content
        cur = self.db.execute(
            "DELETE FROM mem WHERE text LIKE ?", (f"%{query}%",)
        )
        return cur.rowcount


def run_with_blobs(adapter, cases):
    """Re-implements GeneratedCase.run but also returns the retrieved blob."""
    per_case = []
    by_cat = defaultdict(lambda: {"pass": 0, "total": 0})
    by_fam = defaultdict(lambda: {"pass": 0, "total": 0})
    t0 = time.perf_counter()
    for c in cases:
        adapter.reset()
        for f in c.setup_facts:
            adapter.inscribe(f)
        for m in c.mutations:
            op = m[0]
            if op == "supersede":
                adapter.supersede(m[1], m[2])
            elif op == "release":
                adapter.release(m[1])
            elif op == "purge":
                adapter.purge(m[1])
        top = adapter.recall_texts(c.final_query, k=10)
        blob = " ".join(top).lower()
        passed = True
        reason = "pass"
        for s in c.must_contain:
            if s.lower() not in blob:
                passed, reason = False, f"missing must_contain: {s!r}"
                break
        if passed:
            for t in c.must_not_contain:
                if t.lower() in blob:
                    passed, reason = False, f"leaked must_not_contain: {t!r}"
                    break
        cat = case_to_attack_category(c.id)
        by_cat[cat]["total"] += 1
        by_fam[c.family]["total"] += 1
        if passed:
            by_cat[cat]["pass"] += 1
            by_fam[c.family]["pass"] += 1
        per_case.append({
            "id": c.id, "category": cat, "family": c.family,
            "passed": passed, "reason": reason,
            "must_contain": c.must_contain,
            "must_not_contain": c.must_not_contain,
            "final_query": c.final_query,
            "retrieved_blob": " ".join(top),
        })
    wall = time.perf_counter() - t0
    return per_case, dict(by_cat), dict(by_fam), wall


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    adapter = NaiveSQLAdapter()
    per_case, by_cat, by_fam, wall = run_with_blobs(adapter, ADVERSARIAL_TESTS)
    total = len(per_case)
    passed = sum(1 for r in per_case if r["passed"])

    print(f"\n=== Naive-SQL baseline on {total} ForgetEval-Adv cases ===")
    print(f"OVERALL  {passed}/{total} = {passed/total:.1%}   ({wall:.1f}s)\n")
    print(f"{'family':<16}{'pass':>6}/{'tot':<5} rate")
    for fam in ("supersession", "decay", "amnesia", "purge", "drift"):
        d = by_fam.get(fam, {"pass": 0, "total": 0})
        r = d["pass"] / max(d["total"], 1)
        print(f"{fam:<16}{d['pass']:>6}/{d['total']:<5} {r:.0%}")
    print(f"\n{'attack_category':<26}{'pass':>6}/{'tot':<5} rate")
    for cat in sorted(by_cat):
        d = by_cat[cat]
        r = d["pass"] / max(d["total"], 1)
        print(f"{cat:<26}{d['pass']:>6}/{d['total']:<5} {r:.0%}")

    out = {
        "adapter": "naive-sql",
        "suite": "adversarial-385",
        "overall_pass": passed, "overall_total": total,
        "overall_rate": passed / total,
        "by_family": by_fam, "by_category": by_cat,
        "wall_seconds": wall,
    }
    (OUT / "adversarial_results_naive_sql.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "naive_sql_per_case.json").write_text(
        json.dumps(per_case, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote data/adversarial_results_naive_sql.json + "
          f"naive_sql_per_case.json")


if __name__ == "__main__":
    main()

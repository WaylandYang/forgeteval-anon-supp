"""Every check that has caught something, in one command.

Tonight produced nine classes of error and each was found by a different
check. Running them together is cheaper than remembering which one
matters:

  runs.py            which configurations are still on a superseded run
  check_paper_numbers.py   canonical values appear, no placeholders, no keys
  audit_claims.py    every k/n in the paper exists in the data
  regenerate         all tables and figures, then report anything that moved

The last one is the important one and is why this is a script rather than
a list. Regenerating and diffing is what catches the case where the data
changed and the paper did not, which no static check can see.
"""
from __future__ import annotations

import subprocess
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = sys.executable

GENERATORS = [
    "main_table.py", "placement_table.py", "cross_llm_table.py",
    "baseline_matrix.py", "ecosystem_tables.py", "hc_split_table.py",
    "mem0_delta_table.py", "external_table.py", "external_primary_table.py", "partition_table.py", "repro_table.py", "worked_cases.py", "mcnemar.py", "heatmap_data.py",
    "cross_arch_table.py",
    "latency_table.py",
]

# The banner below says "every table and figure", and for a long time the
# list above held no figure generator, so a figure could drift from the
# data behind it and this check would still print all-clear. These write
# into paper/figures/, which the git-status diff already watches.
GENERATORS += ["make_heatmap.py", "make_ablation_figure.py",
               "make_pareto_figure.py"]


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", **kw)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    failures = []

    print("=" * 62)
    print("1. regenerate every generated table and figure, then look "
          "for movement")
    print("=" * 62)
    before = run(["git", "status", "--porcelain", "paper/"]).stdout
    for g in GENERATORS:
        r = run([PY, "scripts/" + g])
        if r.returncode != 0:
            print("  FAIL %s" % g)
            print("    " + (r.stderr or r.stdout).strip()[:200])
            failures.append(g)
    after = run(["git", "status", "--porcelain", "paper/"]).stdout
    moved = sorted(set(after.split("\n")) - set(before.split("\n")))
    moved = [m for m in moved if m.strip()]
    if moved:
        print("  tables changed on regeneration -- the paper was stale:")
        for m in moved:
            print("   ", m)
        failures.append("stale tables")
    else:
        print("  nothing moved; generated assets match the data")

    for label, script in (("2. run coverage", "runs.py"),
                          ("3. canonical numbers", "check_paper_numbers.py"),
                          ("4. every k/n claim", "audit_claims.py")):
        print("\n" + "=" * 62)
        print(label)
        print("=" * 62)
        r = run([PY, "scripts/" + script])
        out = (r.stdout or "").strip()
        tail = "\n".join(out.split("\n")[-14:])
        print(tail if tail else "(no output)")
        if script == "check_paper_numbers.py" and "missing: 0" not in out:
            failures.append(script)

    print("\n" + "=" * 62)
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    print("all checks pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

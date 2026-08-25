"""Audit main-body section sizes to identify compression targets."""
from pathlib import Path
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
content = open(
    str(Path(__file__).resolve().parent.parent / "paper" / "paper.tex"),
    encoding="utf-8",
).read()

main_end = content.find(r"\bibliography")
main = content[:main_end]

# Find all section / subsection / section* markers
markers = []
for m in re.finditer(r"\\(sub)?section\*?\{([^}]+)\}", main):
    markers.append((m.start(), m.group(1) or "", m.group(2)))

# Compute section sizes by adjacent positions
sections = []
for i, (pos, kind, name) in enumerate(markers):
    end = markers[i + 1][0] if i + 1 < len(markers) else len(main)
    sections.append((kind, name, end - pos))

# Add preamble
preamble_size = markers[0][0] if markers else 0
print(f"Total main body: {len(main)} chars (~{len(main)//6} words)\n")
print(f"  {'kind':<6} {'name':<50} {'chars':>6}  bar")
print("-" * 78)
print(f"  {'PRE':<6} {'(preamble, abstract)':<50} {preamble_size:>6}  " +
      "#" * (preamble_size // 200))
for kind, name, size in sections:
    bar = "#" * (size // 200)
    label = kind if kind else "sec"
    print(f"  {label:<6} {name[:48]:<50} {size:>6}  {bar}")

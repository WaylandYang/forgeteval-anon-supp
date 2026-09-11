"""Check bare system-name uses vs the \\sys* macros."""
import re, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

text = Path("paper/paper.tex").read_text(encoding="utf-8")

SYSTEMS = {
    "Lethe":      r"\sysLethe",
    "Mem0":       r"\sysMem",
    "A-MEM":      r"\sysAmem",
    "MemPalace":  r"\sysPalace",
    "ForgetEval": r"\sysForget",
}

for name, macro in SYSTEMS.items():
    # bare uses NOT preceded by backslash or word char (ie, real word)
    bare_pat = re.compile(rf"(?<![\\\\w]){re.escape(name)}\b")
    macro_count = text.count(macro)
    bare_hits = []
    for m in bare_pat.finditer(text):
        # Skip if part of macro definition line
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        line = text[line_start:line_end]
        if "newcommand" in line or "% " in line[:m.start() - line_start]:
            continue
        bare_hits.append(m)
    print(f"{name:12s}  macro: {macro_count:3d}  bare: {len(bare_hits):3d}")
    main_bare = []
    for m in bare_hits:
        line_no = text[:m.start()].count("\n") + 1
        if 50 <= line_no <= 1100:
            main_bare.append((line_no, m))
    if main_bare:
        print(f"  bare in main body (lines 50-1100): {len(main_bare)}")
        for line_no, m in main_bare[:6]:
            ctx = text[max(0, m.start() - 35):m.end() + 35].replace("\n", " | ")
            print(f"    L{line_no}: ...{ctx}...")

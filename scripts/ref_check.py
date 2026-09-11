"""Check label/ref consistency in paper.tex."""
import re, sys
sys.stdout.reconfigure(encoding="utf-8")

text = open("paper/paper.tex", encoding="utf-8").read()
labels = set(re.findall(r"\\label\{([^}]+)\}", text))
refs = set()
for m in re.finditer(r"\\(?:ref|autoref|eqref|nameref)\{([^}]+)\}", text):
    refs.add(m.group(1))
print(f"labels defined: {len(labels)}")
print(f"refs used: {len(refs)}")
broken = refs - labels
orphan = labels - refs
print(f"BROKEN refs (used but not defined): {sorted(broken)}")
print(f"orphan labels (defined but not referenced): {sorted(orphan)}")

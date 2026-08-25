"""Count figures and tables in paper.tex with locations."""
import re, sys
sys.stdout.reconfigure(encoding="utf-8")

text = open("paper/paper.tex", encoding="utf-8").read()

# Figures
print("=== Figures ===")
for m in re.finditer(r"\\begin\{figure\*?\}.{0,400}?\\label\{(fig:[^}]+)\}", text, re.DOTALL):
    line = text[:m.start()].count("\n") + 1
    print(f"  L{line}: {m.group(1)}")

print()
print("=== Tables ===")
for m in re.finditer(r"\\begin\{table\*?\}.{0,500}?\\label\{(tab:[^}]+)\}", text, re.DOTALL):
    line = text[:m.start()].count("\n") + 1
    print(f"  L{line}: {m.group(1)}")

figs = re.findall(r"\\begin\{figure\*?\}", text)
tabs = re.findall(r"\\begin\{table\*?\}", text)
print()
print(f"Total: {len(figs)} figures, {len(tabs)} tables")

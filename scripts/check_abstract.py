"""Quick abstract word-count and self-review for the paper."""
from pathlib import Path
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

text = open(
    str(Path(__file__).resolve().parent.parent / "paper" / "paper.tex"),
    encoding="utf-8",
).read()

# Find abstract
start = text.find(r"\begin{abstract}")
end = text.find(r"\end{abstract}")
if start >= 0 and end >= 0:
    abs_text = text[start + len(r"\begin{abstract}"):end]
    # Strip LaTeX
    clean = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})*", " ", abs_text)
    clean = re.sub(r"[{}]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    words = clean.split()
    print(f"Abstract: {len(words)} words")
    print(f"Standard ACL abstract: ~150-200 words")
    print(f"Over budget by: {max(0, len(words) - 200)} words")

# Main body word count
ref_marker = text.find(r"\bibliography")
main = text[:ref_marker]
clean = re.sub(r"%[^\n]*\n", "\n", main)
clean = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})*", " ", clean)
clean = re.sub(r"[{}]", " ", clean)
clean = re.sub(r"\s+", " ", clean).strip()
print(f"\nMain body: ~{len(clean.split())} words")

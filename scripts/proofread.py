"""Pattern-based grammar/format proofread for paper.tex."""
import re, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

text = Path("paper/paper.tex").read_text(encoding="utf-8")

CHECKS = [
    (r"\ba\s+([aeiouAEIOU][a-z])", '"a" before vowel'),
    (r"\ban\s+([bcdfghjklmnpqrstvwxyz][a-z])", '"an" before consonant'),
    (r" ,", "space before comma"),
    (r" \.", "space before period"),
    (r",,", "double comma"),
    (r"\.\.[a-z]", "double period before lowercase"),
    (r"\bthe the\b", '"the the"'),
    (r"\bof of\b", '"of of"'),
    (r"\bin in\b", '"in in"'),
    (r"\band and\b", '"and and"'),
    (r"\bis is\b", '"is is"'),
    (r"\bteh\b", '"teh" typo'),
    (r"\brecieve", '"recieve" typo'),
    (r"\bseperate", '"seperate" typo'),
    (r"\boccured\b", '"occured" typo'),
    (r"\bparamter", '"paramter" typo'),
    (r"\bbencmark", '"bencmark" typo'),
    (r"\bsupercede", '"supercede" (use supersede)'),
    (r"\bagainst against\b", "double against"),
    (r"\(\s+\)", "empty parens"),
    (r"\\textbf\{\s*\}", "empty textbf"),
    (r"\\emph\{\s*\}", "empty emph"),
    (r"\\\\\s*\\\\", "double row break (\\\\\\\\)"),
    (r"\\textit\{[A-Z][a-z]+\s+[A-Z][a-z]+\}", "italicized name pair (potential author)"),
    (r"\bMem0\+v3\b", "Mem0+v3 (count)"),
    (r"\bM0v3\b", "M0v3 short form (count)"),
    (r"\bOpenMemory\b", "OpenMemory (count)"),
    (r"\bOpenMem\b", "OpenMem short form (count)"),
    (r"\bHippoRAG\b", "HippoRAG (count)"),
    (r"\bHippo\b", "Hippo short form (count)"),
    (r"\\sysLethe\{\}", "\\sysLethe usage (count)"),
    (r"\bLethe\b(?!\s*\\)", "bare Lethe (should use \\sysLethe)"),
    (r"\bLangGraph\+LLM\b", "LangGraph+LLM (count)"),
    (r"\bLG\+LLM\b", "LG+LLM short (count)"),
    (r"\$\\\\sim\\$\\\\", "\\\\sim macro check"),
    (r"\\sim\\$", "\\\\sim followed by $"),
    # Hyphenation consistency
    (r"\bin house\b", '"in house" (should be hyphenated "in-house")'),
    (r"\bcross lingual\b", '"cross lingual" (should be hyphenated)'),
    (r"\bopen source\b", "'open source' (check if should be hyphenated)"),
    # Trailing whitespace at end of line
    (r" $", "trailing whitespace"),
]

found_any = False
for pat, desc in CHECKS:
    matches = list(re.finditer(pat, text, re.MULTILINE))
    if not matches:
        continue
    if "(count)" in desc:
        print(f"  [{desc}] {len(matches)} occurrences (review for consistency)")
        continue
    found_any = True
    print(f"\n[{desc}] {len(matches)} hits:")
    for m in matches[:8]:
        line_no = text[:m.start()].count("\n") + 1
        ctx = text[max(0, m.start() - 40):m.end() + 40].replace("\n", " | ")
        print(f"  L{line_no}: ...{ctx}...")

if not found_any:
    print("\n[no grammar/format issues detected]")

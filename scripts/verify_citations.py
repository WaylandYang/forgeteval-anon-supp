"""Verify every bibliography entry against its primary source.

A citation can be wrong in three ways that a build never catches: the
entry can be defined but never cited, cited but never defined, or defined
with metadata that does not match the work it points at -- a title
attached to the wrong authors, a plausible-looking arXiv id for a paper
that does not exist. The third kind is the one that survives proofreading,
and the one an LLM-assisted workflow is most likely to introduce.

This script checks all three. For entries carrying an arXiv id it fetches
the record from the arXiv API and compares title and author list against
what the bib file claims, reporting any drift rather than silently
"fixing" it: a mismatch may mean the id is wrong, or the title is, and
only a human can say which.

  python scripts/verify_citations.py

Exit status is non-zero if any entry fails, so it can gate a build.
"""
from __future__ import annotations

import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIB = ROOT / "paper" / "refs.bib"
TEX = ROOT / "paper" / "paper.tex"
API = "http://export.arxiv.org/api/query?id_list="
ATOM = "{http://www.w3.org/2005/Atom}"


def parse_bib(text):
    entries = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),", text):
        key = m.group(2).strip()
        start = m.end()
        depth, i = 1, m.start() + text[m.start():].index("{")
        i += 1
        while i < len(text) and depth:
            depth += (text[i] == "{") - (text[i] == "}")
            i += 1
        body = text[start:i]
        fields = {}
        for f in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*(?=\w+\s*=|\Z)",
                             body, re.S):
            fields[f.group(1).lower()] = " ".join(f.group(2).split())
        entries[key] = fields
    return entries


ACCENTS = {"\\'": "", '\\"': "", "\\`": "", "\\^": "", "\\~": "",
           "\\c": "", "\\v": "", "\\=": "", "\\.": ""}


def strip_tex(s):
    """Drop LaTeX accent commands and braces so 'Guti{\\'e}rrez' compares
    equal to 'Gutiérrez'."""
    for k in ACCENTS:
        s = s.replace(k, "")
    return re.sub(r"[{}\\]", "", s or "")


def norm(s):
    s = strip_tex(s)
    s = (s.replace("é", "e").replace("è", "e")
          .replace("ü", "u").replace("ö", "o")
          .replace("á", "a").replace("í", "i")
          .replace("ó", "o").replace("ú", "u")
          .replace("ñ", "n").replace("ç", "c"))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def surnames(bib_author):
    """Surnames from a BibTeX author field, in order."""
    out = []
    for part in re.split(r"\s+and\s+", bib_author or ""):
        part = part.strip()
        if not part:
            continue
        out.append(part.split(",")[0].strip() if "," in part
                   else part.split()[-1])
    return out


def fetch_arxiv(arxiv_id, tries=3):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(API + arxiv_id, timeout=30) as r:
                root = ET.fromstring(r.read())
            break
        except Exception:
            if attempt == tries - 1:
                return "ERROR"
            time.sleep(5 * (attempt + 1))
    entry = root.find(ATOM + "entry")
    if entry is None:
        return None
    title = " ".join(entry.findtext(ATOM + "title", "").split())
    authors = [a.findtext(ATOM + "name", "")
               for a in entry.findall(ATOM + "author")]
    # a withdrawn/nonexistent id comes back as an entry with no id
    if not title or "Error" in title:
        return None
    return {"title": title, "authors": authors}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    bib = parse_bib(BIB.read_text(encoding="utf-8-sig"))
    tex = TEX.read_text(encoding="utf-8")

    cited = set()
    for m in re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]+)\}", tex):
        cited |= {k.strip() for k in m.group(1).split(",")}

    print(f"bib entries {len(bib)}   cite keys used {len(cited)}\n")

    undefined = sorted(cited - set(bib))
    uncited = sorted(set(bib) - cited)
    if undefined:
        print(f"!! CITED BUT NOT DEFINED: {undefined}")
    if uncited:
        print(f"   defined but never cited: {uncited}")
    print()

    problems = list(undefined)
    checked = 0
    for key, f in sorted(bib.items()):
        aid = f.get("eprint") or ""
        if not aid:
            m = re.search(r"arxiv\.org/abs/([0-9.]+)", f.get("url", ""))
            aid = m.group(1) if m else ""
        if not aid:
            continue
        checked += 1
        meta = fetch_arxiv(aid)
        time.sleep(3)  # arXiv asks for one request every 3 seconds
        if meta == "ERROR":
            print(f" ? {key:<16} arXiv {aid}: fetch failed, unchecked")
            continue
        if meta is None:
            print(f"!! {key:<16} arXiv {aid}: NO SUCH RECORD")
            problems.append(key)
            continue

        bib_title = re.sub(r"[{}]", "", f.get("title", ""))
        t_ok = norm(bib_title) == norm(meta["title"])
        want = [norm(s) for s in surnames(f.get("author", ""))]
        got = [norm(a.split()[-1]) for a in meta["authors"]]
        # "and others" is an explicit et al.: check the prefix it spells out
        if want and want[-1] == "others":
            want = want[:-1]
            got = got[:len(want)]
        a_ok = want == got

        if t_ok and a_ok:
            print(f"   {key:<16} arXiv {aid}  ok")
        else:
            print(f"!! {key:<16} arXiv {aid}")
            if not t_ok:
                print(f"      bib title : {bib_title}")
                print(f"      arXiv     : {meta['title']}")
            if not a_ok:
                print(f"      bib authors : {'; '.join(surnames(f.get('author','')))}")
                print(f"      arXiv       : {'; '.join(meta['authors'])}")
            problems.append(key)

    print(f"\narXiv entries checked {checked}; "
          f"entries without a verifiable id {len(bib) - checked}")
    print(f"problems: {len(problems)}"
          + (f"  -> {sorted(set(problems))}" if problems else ""))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

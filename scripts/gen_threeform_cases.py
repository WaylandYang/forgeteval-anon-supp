"""Generate three-surface-form cross-lingual cases.

The externally-authored subset exposed a limit: its cross_lingual cases
give an entity three surface forms where ours give two, and the hook
scores 0/8 on them against 32/38 in house. The paper currently reads the
in-house figure as an upper bound for two-form cases and leaves it there,
which concedes the point without measuring it.

This builds the missing rung. Same structure as the in-house cases --
target entity, a sibling that must survive, a purge naming one form --
with a third form added for the target and for the sibling, so the only
thing that changes between the two-form and three-form conditions is how
many surfaces of one entity have to be reached.

Forms per script family: native, romanised, and a third that is either a
second romanisation, an initialised form, or a common transliteration
variant -- all attested ways the same entity appears in real records.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# (native, romanised, third form, script label)
TARGETS = [
    ("张伟", "Zhang Wei", "Chang Wei", "Han"),
    ("李娜", "Li Na", "Lee Na", "Han"),
    ("田中太郎", "Tanaka Taro", "Taro Tanaka", "Kana"),
    ("Дмитрий Иванов", "Dmitry Ivanov", "Dmitrii Ivanov", "Cyrillic"),
    ("Γιώργος Παπάς", "Giorgos Papas", "Yorgos Papas", "Greek"),
    ("김민준", "Kim Minjun", "Kim Min-jun", "Hangul"),
    ("محمد الأمين", "Mohamed Al-Amin", "Muhammad Alamin", "Arabic"),
    ("आरव शर्मा", "Aarav Sharma", "Arav Sharma", "Devanagari"),
]

# One sibling per case, same script family, must survive.
SIBLINGS = [
    ("王芳", "Wang Fang", "Wong Fang"),
    ("刘洋", "Liu Yang", "Lau Yang"),
    ("鈴木花子", "Suzuki Hanako", "Hanako Suzuki"),
    ("Ольга Петрова", "Olga Petrova", "Olha Petrova"),
    ("Ελένη Νικολάου", "Eleni Nikolaou", "Helene Nikolaou"),
    ("박지훈", "Park Jihoon", "Park Ji-hoon"),
    ("فاطمة الزهراء", "Fatima Al-Zahra", "Fatma Alzahra"),
    ("प्रिया वर्मा", "Priya Verma", "Priya Varma"),
]


def build(n_forms: int):
    """Cases where the target appears under `n_forms` surface forms."""
    cases = []
    for i, ((t1, t2, t3, script), (s1, s2, s3)) in enumerate(
            zip(TARGETS, SIBLINGS), 1):
        tf = [t1, t2, t3][:n_forms]
        sf = [s1, s2, s3][:n_forms]

        facts = [f"Customer {tf[0]} placed an order in March."]
        if n_forms > 1:
            facts.append(f"The same customer in romanised form: {tf[1]} "
                         f"subscribed to Plan A.")
        if n_forms > 2:
            facts.append(f"Billing records list the same customer as "
                         f"{tf[2]}, invoice 4471.")
        facts.append("Marketing team rotated coupons last quarter.")
        facts.append(f"Customer {sf[0]} opened a separate account in April.")
        if n_forms > 1:
            facts.append(f"The same customer, written {sf[1]}, is on the "
                         f"Premium tier.")
        if n_forms > 2:
            facts.append(f"Support notes record {sf[2]} as the contact.")

        cases.append({
            "id": "adv_cross_lingual_%dform_%02d" % (n_forms, i),
            "category": "cross_lingual_identifier",
            "script": script,
            "n_forms": n_forms,
            "setup_facts": facts,
            # The request names exactly one form, as the suite requires.
            "mutations": [["purge", "customer %s" % tf[1]]],
            "final_query": "List every customer on file, including %s."
                           % sf[1],
            # The sibling survives; every form of the target must go.
            "must_contain": [sf[1]],
            "must_not_contain": tf,
        })
    return cases



# A third condition the external subset actually tests, which the
# form-count conditions do not: a historical exonym shares no morphology
# with the current name, so no rule over the string reaches it and the
# model has to know the fact. This is what "Peking" is to "Beijing", and
# it is what the external cross_lingual cases turn on.
EXONYMS = [
    ("Beijing", "北京", "Peking", "Han"),
    ("Guangzhou", "广州", "Canton", "Han"),
    ("Mumbai", "मुंबई", "Bombay", "Devanagari"),
    ("Yangon", "ရန်ကုန်", "Rangoon", "Burmese"),
    ("Chennai", "சென்னை", "Madras", "Tamil"),
    ("Kolkata", "কলকাতা", "Calcutta", "Bengali"),
    ("Istanbul", "İstanbul", "Constantinople", "Latin"),
    ("Thessaloniki", "Θεσσαλονίκη", "Salonica", "Greek"),
]


def build_exonym():
    """Same shape, but the third form is an exonym rather than a variant."""
    cases = []
    for i, ((cur, native, old, script), (s1, s2, _)) in enumerate(
            zip(EXONYMS, SIBLINGS), 1):
        cases.append({
            "id": "adv_cross_lingual_exonym_%02d" % i,
            "category": "cross_lingual_identifier",
            "script": script,
            "n_forms": 3,
            "setup_facts": [
                f"The regional operations center is located in {cur}.",
                f"Internal documents refer to the city as {native}.",
                f"Older logistics records still use the spelling {old}.",
                "The compliance audit finished in April.",
                f"Customer {s1} opened a separate account in April.",
                f"The same customer, written {s2}, is on the Premium tier.",
            ],
            "mutations": [["purge", "%s regional operations center" % cur]],
            "final_query": "List every customer on file, including %s." % s2,
            "must_contain": [s2],
            "must_not_contain": [cur, native, old],
        })
    return cases


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out = {"note": "Two- and three-form cross-lingual cases, identical in "
                   "every respect but the number of surface forms the "
                   "target entity carries.",
           "two_form": build(2), "three_form": build(3),
           "exonym": build_exonym()}
    dest = ROOT / "data" / "threeform_cases.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print("wrote %s: %d two-form, %d three-form, %d exonym"
          % (dest.name, len(out["two_form"]), len(out["three_form"]),
             len(out["exonym"])))
    print("scripts:", ", ".join(sorted({t[3] for t in TARGETS})))


if __name__ == "__main__":
    main()

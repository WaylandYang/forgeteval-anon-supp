"""One place that decides which result file is authoritative.

Every LLM configuration in this paper was first measured with the
completion budget capped at 512 tokens. Responses longer than that came
back truncated or empty, and an empty response falls through to the
deterministic path, so the shortfall was invisible: the runs completed,
reported no errors, and produced plausible numbers. The prompts that
overran are the ones that have to emit rewritten row text, which is why
the loss concentrated in compound_fact and in non-Latin canonicalization
rather than spreading evenly.

Re-measured runs carry the RETAG suffix. Rather than edit every
generator each time one lands, they all call resolve() and get the
re-measured file when it exists and the original when it does not. A
mixed table is therefore impossible to produce by forgetting to update
something -- the only way to get an old number is for no new file to
exist, and audit() lists exactly those.
"""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RETAG = "_mt3000"

# The hooked reference configuration is reported from the run at the mean
# of five independent executions (348, 344, 342, 338, 338), not from the
# best of them. Everything derived from its per-case verdicts -- the
# paired tests, the category breakdown, Table 2's headline row -- comes
# from that one run so the numbers are mutually consistent.
DESIGNATED = {
    "openrouter_hook_deepseek_deepseek-v4-flash_v07_probed.json":
        "openrouter_hook_deepseek_deepseek-v4-flash_v07_probed_mt3000repr3.json",
}


def resolve(name: str) -> pathlib.Path:
    """Path to the authoritative result file for `name`.

    `name` is the original (pre-re-measurement) file name. Returns the
    re-measured path when it exists, otherwise the original.
    """
    if name in DESIGNATED:
        d = DATA / DESIGNATED[name]
        if d.exists():
            return d
    p = DATA / name
    if name.endswith(".json"):
        rp = DATA / (name[:-5] + RETAG + ".json")
        if rp.exists():
            return rp
    return p


# Re-run under the wider budget by a script that overwrites its own
# output rather than tagging it, so no _mt3000 file exists to detect.
# Verified by the run itself, not by a filename.
REMEASURED_IN_PLACE = {
    "external_v07_lethe_llm.json",
    "ablate_v0_original_v07_probed.json",
    "ablate_v1_zeroshot_v07_probed.json",
    "ablate_v2_reworded_v07_probed.json",
    "nli_blobs_lethe_llm_v07_summary.json",
}


def is_remeasured(name: str) -> bool:
    if name in REMEASURED_IN_PLACE or name in DESIGNATED:
        return True
    return resolve(name).name.endswith(RETAG + ".json")


def audit(names) -> list[str]:
    """Which of these are still on the capped measurement."""
    return [n for n in names if not is_remeasured(n)]


if __name__ == "__main__":
    import sys

    P = "openrouter_hook_deepseek_deepseek-v4-flash_"
    LLM_RUNS = [
        P + "v07_probed.json",
        P + "langgraph_v07_probed.json",
        P + "routed_v07_probed.json",
        P + "routed-langgraph_v07_probed.json",
        P + "mem0-infer_v07_probed.json",
        P + "inscribe_v07_probed.json",
        P + "inscribe-aware_v07_probed.json",
        P + "merge-inscribe_v07_probed.json",
        P + "inscribe+mutation_v07_probed.json",
        "openrouter_hook_openai_gpt-55_v07_probed.json",
        "openrouter_hook_anthropic_claude-opus-48_v07_probed.json",
        "openrouter_hook_google_gemini-31-pro-preview_v07_probed.json",
        "openrouter_hook_qwen_qwen36-max-preview_v07_probed.json",
        "openrouter_hook_deepseek_deepseek-v4-pro_v07_probed.json",
        "external_v07_lethe_llm.json",
        "ablate_v0_original_v07_probed.json",
        "ablate_v1_zeroshot_v07_probed.json",
        "ablate_v2_reworded_v07_probed.json",
        "nli_blobs_lethe_llm_v07_summary.json",
        # The requirements table reads the hooked configuration at each
        # requirement level, so its v05 and v06 runs are load-bearing
        # too. They were absent from this list, which is why it could
        # report every run re-measured while that table still mixed a
        # capped column with an uncapped one.
        P.rstrip("_") + ".json",
        P + "v06.json",
    ]
    pending = audit(LLM_RUNS)
    done = len(LLM_RUNS) - len(pending)
    print("re-measured %d/%d LLM runs" % (done, len(LLM_RUNS)))
    for n in pending:
        print("  PENDING  " + n)
    sys.exit(0)

# Agent Memory Can Only Forget What Its Deletion Path Can See

Anonymous supplement for a paper under double-blind review. It contains the
ForgetEval-Adv benchmark, every adapter evaluated in the paper, the Lethe
reference store, and the scripts that generate each table and figure. MIT
licensed.

The manuscript itself is not in this repository; it is on the review site.

## Layout

| path | what is in it |
| --- | --- |
| `bench/forgeteval/` | the evaluation protocol, the 385 adversarial cases, the Adapter Protocol, and the primary adapters |
| `lethe/` | the reference memory store, including the control plane (`supersede`, `release`, `purge`) |
| `scripts/` | experiment runners, plus one generator per table and figure |
| `data/` | per-case verdicts and aggregates for every run the paper reports |
| `iaa/` | annotation instructions and the 100 sampled cases behind the agreement numbers |
| `recipes/` | five worked examples of the control plane |
| `docs/` | protocol notes for ForgetEval and its adversarial extension |
| `tests/` | store-level tests |

## Reproducing the paper's numbers

```
python scripts/verify_all.py
```

This regenerates every table and figure from `data/`, fails if any of them
moves, checks that each reported run was measured under the current token
limit, and re-derives every `k/n` claim in the manuscript. It reads no
network and needs no key.

`scripts/runs.py` is the single resolver from a run's name to the file that
is authoritative for it. Anything reading results should go through it
rather than naming a file directly, which is how two tables once ended up
quoting a superseded measurement.

## What runs without a key

The four primary adapters are deterministic. They need no API key, no GPU,
and no network:

```
python scripts/run_openrouter_hook.py --no-llm --adapter lethe
```

Embeddings are computed locally (`fastembed`, ONNX, 384-d).

The LLM-hook configurations need an OpenAI-compatible endpoint, supplied
through `OPENROUTER_API_KEY` or `LLM_API_KEY`. The extended ecosystem
adapters additionally need whatever service they wrap -- Letta,
Graphiti/Neo4j, OpenMemory, Docker. Service URLs come from the environment
and are never hardcoded.

A run that requests an LLM and never reaches one refuses to write its
results rather than reporting the deterministic score as a measurement.

## Not included

The raw files from the external contributors are not redistributed. The
parsed and admitted cases derived from them ship as
`data/external_subset_cases.json`, which is what the tables read.

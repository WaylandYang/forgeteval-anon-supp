# ForgetEval-Adv — IAA Instructions (multi-annotator)

Thanks for helping validate the ForgetEval-Adv benchmark.

## Background

ForgetEval-Adv is a 400-case benchmark for testing whether LLM-agent
memory systems correctly **forget** facts when told to. Each case has
explicit `must_contain` (should remain after mutations) and
`must_not_contain` (should NOT remain after mutations) assertions.

We use an independent LLM judge for case admission; we are asking
**NLP/CS-trained annotators** like you to independently confirm
admission decisions on a 100-case stratified sample (10 cases per
attack category, mix of hand-crafted v0.4 and LLM-drafted v0.5).

**Your task: 100 binary judgements + 1 short reason per case.**
Estimated time: **90–120 minutes**.  Feel free to break it into
2 sittings; just pace yourself so judgment quality stays consistent.

## What "well-formed" means

For each case, simulate by hand:

1. **START STATE**: the `setup_facts` are written into memory.
2. **APPLY MUTATIONS** in order:
   - `supersede(q, new)`: replace the row best-matching `q` with `new`.
   - `release(q)`: mark rows matching `q` as unreachable.
   - `purge(q)`: hard-delete rows matching identifier `q`. For
     `identifier_obfuscation` / `cross_lingual_identifier`
     categories, purge is canonicalizing — different surface forms
     of the same identifier are all removed.
3. **END STATE**: the rows still reachable for retrieval.

A case is **well-formed** iff:
- (a) every string in `must_contain` is literally a substring of
  some surviving row, **and**
- (b) no string in `must_not_contain` is a substring of any
  surviving row or of any `must_contain` string.

Mark `wf` if both (a) and (b) hold. Mark `ill` if either fails.

## Common ill-formed patterns to reject

1. **Substring collision**: a `must_not_contain` string is a
   substring of a surviving `must_contain` string or any surviving
   fact (e.g. forbidding "TXN-123" while the surviving fact is
   "TXN-12345").
2. **Self-substring trap**: a `must_not_contain` substring
   literally appears in a setup fact that isn't being removed
   by any mutation.
3. **Phantom must_contain**: the only fact mentioning a
   `must_contain` string is one that gets superseded / released /
   purged.
4. **Ambiguous query**: the `final_query` can't be answered from
   the end state.

## How to record your decisions

1. Open the supplied `cases.csv` file in Excel/Numbers/LibreOffice.
2. For each row, fill in the `your_verdict` column with either
   `wf` or `ill` (no spaces, lowercase).
3. Fill in `your_reason` with one short sentence when you mark
   `ill`. Leave blank for `wf`.
4. Optionally fill in `time_seconds` to record how long the case
   took (helps us calibrate; not required).
5. Save the file as `<your_id>.csv` (e.g. `annotator_03.csv` —
   any unique identifier works; we anonymize before computing
   stats).
6. Email the saved file back to the requester.

**Important**: do NOT confer with other annotators before
submitting. The whole point is independent judgements.

## What we do with your labels

We compute multi-annotator agreement metrics:

- **Fleiss' kappa** across all annotators
- **Per-case agreement distribution** (how many annotators agree)
- **Majority-vote consensus** vs. our LLM-judge admission verdict
- **Per-category kappa** to see which categories have less
  human–machine agreement

Your identity is NOT linked to your labels in the paper. We
acknowledge contributions in aggregate (e.g. "10 NLP/CS-trained
annotators independently labeled..."). If you'd like to be named
in the Acknowledgments section explicitly, mention it when you
return your CSV; otherwise we default to anonymous aggregate
acknowledgement.

## Edge cases

- **A case feels too easy**: still mark `wf`. Easy is valid.
- **A case feels too hard for any system**: still mark `wf` if
  it's logically consistent. "Hard" is not "ill-formed".
- **You're genuinely unsure**: mark `ill` with reason "unclear".
  We expect ~5% disagreement; uncertainty is a valid signal.

## Questions

Reply to the requester. Thanks!

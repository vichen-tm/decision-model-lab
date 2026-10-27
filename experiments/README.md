# Evidence navigation

The primary research narrative is in [experiment results](../docs/experiment-results.md). This directory retains machine-readable evidence, including unsuccessful experiments.

| Campaign | Run plans | Unique response records | Layout |
|---|---:|---:|---|
| AG News | 33 | 25,455 | Baseline/recovery; rules rounds 1–3; fresh rules; universal search; wide G/J/K/L; model-specific wide checks; Jev category-boundary revision |
| ProofWriter | 32 | 10,627 | Jev unknown v1/v2 and controls; broad Jev; Liquid transfer/repair and broad checks; Microsoft transfer/broad checks; independent final validation |

Run-plan counts include recovery plans, not just new hypotheses. `runs.csv` lists planned task counts, observed responses and newly retained unique records. `tasks.jsonl.gz` records membership even when a recovery reused an earlier response. `attempt-ledger.jsonl.gz` preserves intents and whether a response exists. Copied intents/responses across recovery plans describe the same execution and must not be summed as extra charges.

`responses.jsonl.gz` deduplicates the same task plus canonical raw response. It includes technical errors. `rescored-responses.csv` assigns each unique response to its first recorded run; do not use its rows alone to infer that a recovery run has no accepted work. `cost-and-response-summary.json` gives deduplicated known charges and unknown-response costs. Missing-response intents remain in the ledger; their actual billing is unknown.

For each response:
1. `case_sha256` links to `corpus-index.json`, including original gold/source and the case skeleton without input state.
2. `request_template_sha256` links to the exact questions/model/provider template.
3. `request_body_sha256` identifies the complete original request bytes.
4. `raw` holds response bytes/text, status, timing and provider usage.

The author's full corpus is preserved locally under ignored `private/`. Public users obtain original text separately and run `tools/restore_inputs.py`. This verifies exact state and full-case hashes before writing a runnable dataset. See [attribution](../ATTRIBUTION.md).

The additional `proofwriter/upstream-baselines.jsonl.gz` contains 4,500 historical English baseline responses for Jev, Liquid and Microsoft (five repeats). These are separate from the 10,627 new response records. They explain small-stage baseline comparisons and stability observations; broad working/final validation uses one response per model and case.

`working-*` and `final-*` are analysis-ready artifacts. The combined table joins two different evaluation regimes and is labeled accordingly. Recompute them with the commands in [quickstart](../docs/quickstart.md).

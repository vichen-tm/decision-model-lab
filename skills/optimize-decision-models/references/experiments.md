# Bounded experiments

Before inference, save the hypothesis, corpus hash, task IDs, pinned model route/version, exact prompt, number of calls, concurrency, price basis and continuation criteria. Use current provider prices; historical prices in this repository are evidence from that campaign only.

Cost estimate = calls × (input tokens × input price + output tokens × output price), with token prices expressed consistently. Add explicitly approved retry/referral allowance. For variable inputs estimate a range from token counts. `decision-lab estimate` is a calculator, not a live price lookup or hard billing cap. Approve a finite stage and monitor actual usage; stop scheduling when it would exceed authorized scope. Local models may still have meaningful compute/time costs.

Use a small diagnostic first. A baseline compares the same examples and gold across candidates. Save every request intent before sending, raw responses, route/version, usage, cost, elapsed time and outcome. The provided OpenRouter runner does this, has no execution deadline, and uses one repeat by default. No automatic retries: after an actual failure inspect `report-run`; authorize a technical retry or resolve unknown/failed tasks explicitly. Rate-limit errors warrant lower concurrency or a pause, not a hidden retry loop.

For error diagnosis compare incorrect cases with matched correct controls. Ask: is the label supported, is context sufficient, is the instruction faithful to task semantics, does a valid reasoning path exist, is the response parser correct? A plausible explanation from an LLM is a hypothesis, not a causal measurement.

Prompt stages: selected failures → preserved correct and boundary examples → broad working data. Record net improvement and class-specific losses. Predeclare a small number of variants and continuation criteria. Initial failure-focused scores are diagnostic and must not be described as representative accuracy. Broad data reused for variant selection are development data.

The API adapter uses OpenRouter Decisions and choice/noul/score request contracts. The general analysis export supports one choice question per case. For a different provider, preserve the same journal semantics through an explicit adapter; do not claim it is supported by the bundled transport.

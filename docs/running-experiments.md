# Running and recovery

The reusable runner comes from the first benchmark's tested core. Each prepared run stores a hashed immutable `plan.json`, a POSIX lock, and append-only attempt records. It writes intent before sending, then the raw response, usage and status. Transport disables redirects and implicit environment proxies, uses no execution timeout, and has no hidden retries. Ctrl+C stops new scheduling and lets in-flight writes finish.

`report-run` distinguishes accepted, pending, technical failures and unknown outcomes. Check actual processes/locks before resuming. `run --send --retry-technical` explicitly retries eligible technical failures, preserving past attempts. It does not resend accepted calls. Unknown outcomes need a deliberate `resolve --state unknown --action retry|abandon --task-id ID --run-dir DIR`; failed validation uses `--state failed`. Resolve only after reviewing the evidence. A retry of an unknown outcome can duplicate an already billed request.

Concurrency is per provider, shared across models in a plan. Start small; persistent 429 overload warrants stopping and waiting. Existing run plans are immutable; prepare a new plan for changed routes or inputs. Do not silently substitute models.

Budget estimates use supplied prices and token assumptions. The runner is bounded by the prepared task count, not a hard dollar ceiling. For strict spend limits use small authorized batches and provider-side spending controls. Summaries must include known failed-attempt charges and explicitly unknown amounts.

Before public packaging scan code, JSON, gzip contents and decoded raw response bytes for secrets and personal paths. Corpus texts may contain benchmark entities; distinguish those from user/account data. Never include `.env`, private files or review CLI transcripts.

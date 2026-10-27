# Verification record

Local verification on 2026-10-27. No new paid benchmark inference was needed for repository verification.

- 23 unit/integration checks: conditional calls and referral accounting, no invented confidence/cost, changed gold rejection, durable intents, explicit retry/recovery, locking, per-provider concurrency, accepted-response reuse, route/scoring failure handling, language-qualified exports and nonzero exit on incomplete runs.
- `tools/verify_evidence.py`: 9,000 raw responses agree with the six working and three final prediction sets, gold, native confidence, probabilities, usage cost and elapsed time.
- `tools/reproduce.py`: every final policy prediction/call mask, final and working counts/costs, and combined counts match recorded results.
- Full working search rerun: all 128 policy/target count comparisons match the original study. Nested-selection JSON matches the original exactly; its independent verifier performs 17,435 checks.
- Source reconstruction: 1,000 final ProofWriter Jev cases and 2,000 AG News Jev baseline cases reconstructed with exact input-state and full-case hashes.
- Skill frontmatter/name/scaffold validation passed. Independent scenario review covered ambiguous labels, a $2 budget and absent confidence without inventing signals or unauthorized spend.
- Package built as a wheel, installed into an isolated virtual environment with preinstalled NumPy/build dependencies, and the installed CLI ran from outside the source directory. Python 3.12 and 3.14 were exercised; 3.10 is the declared minimum, not a separately tested runtime.
- Offline report exports Markdown, JSON, CSV and parseable SVG. Publication scanning covers text, gzip files and decoded response bytes. Environment files, private corpora and local archives are excluded by the publication allowlist.

The live OpenRouter transport was tested with mocked responses and validated historical provider records. Current provider availability and a new user's credentials are checked when that user authorizes a new experiment.

## Independent reviews

Round 1: an xhigh reviewer and Claude reviewed in parallel. Fixed all-attempt cost accounting, invalid usage handling, language identity, diagnostic/final separation, incomplete-run exit status, frozen model bindings, null groups and evidence references. Removed an unused old CLI.

Round 2: both reviewers found no remaining code blockers. The xhigh reviewer verified exclusion of environment files and old archives. Claude confirmed the corrected bindings, grouping, export and portable reproduction tools; its only publication note was to regenerate the manifest after concurrent final edits. The final manifest and archive are rebuilt after this document. Claude used static review; the parent executed the checks listed above.

## Publication privacy review

Eight AG News technical error bodies contained a provider account identifier (`user_id`). The public copies omit that field; response text, base64 and hashes agree with the redacted bytes. The attempt ledger points to the published records and retains source hashes. `experiments/ag-news/publication-redactions.json` documents each change. Original bodies remain local. Predictions, status codes, timing and cost fields are unchanged.

# Experiments and results

## AG News: a useful failed direction

We collected a six-model baseline, counted shared errors, then examined failures alongside preserved correct answers. Prompt variants attempted to clarify category boundaries without test-specific examples or arbitrary priorities. Selected-error improvements repeatedly weakened on broader samples. Jev's boundary revision scored 484/618 versus baseline 480, but 469/500 versus baseline 474 on the broad portion. This did not support a general improvement claim.

Inspection exposed overlapping categories, insufficient context and some label disagreements. We could not derive a consistent decision rule for several boundaries. The lesson is to inspect what a benchmark's labels mean for the intended task, rather than optimizing a score without a defensible answer contract. All source run identities, prompt templates and saved responses are indexed in `experiments/ag-news/`; summary artifacts preserve the unsuccessful variants too.

## ProofWriter: explicit answer semantics

The next task supplies facts, rules and a statement. True means derivable, false means its explicit negation is derivable, unknown means neither is established. We tested clarifications of unknown versus false and valid forward derivation. Small error/control stages preceded broad working evaluation.

| Working 1,000 | Baseline | Candidate | Change |
|---|---:|---:|---:|
| Jev | 910 | 963 | +53 / +5.3 percentage points |
| Liquid | 883 | 894 | +11 / +1.1 points |
| Microsoft | 944 | 942 | −2; baseline retained |

Jev and Liquid used different successful appendices. Transferring a prompt between models was not uniformly helpful. These are development outcomes, not independent estimates after model/prompt selection.

## Offline methods

Saved aligned outputs allowed confidence analysis, predicted-category correctness calibration, error-overlap calculations, single-model filtering, two/three-stage cascades, agreement, conditional majority/weighted voting and strict signal rules without new model calls. Calibration and policy selection used working folds; the archive also retains a nested-selection analysis. The four demonstrated policy structures were frozen before opening the final thousand.

Working error histogram for exactly 0/1/2/3 wrong models: **831 / 141 / 26 / 2**. Final: **834 / 116 / 44 / 6**. Overlap ranges are inclusive, so “at least two” includes all three.

## Independent final 1,000

| Policy | Working correct / accepted (OOF) | Final correct / accepted | Final errors | Final referrals | USD / million inputs |
|---|---:|---:|---:|---:|---:|
| Jev | 963/1000 | 955/1000 | 45 | 0 | 32.91 |
| Liquid | 894/1000 | 891/1000 | 109 | 0 | 7.07 |
| Microsoft | 944/1000 | 932/1000 | 68 | 0 | 12.57 |
| Conditional weighted ensemble | 984/1000 | 961/1000 | 39 | 0 | 35.95 |
| Confidence + unanimity | 924/928 | 928/937 | 9 | 63 | 35.95 |
| Microsoft → Jev | 912/920 | 923/929 | 6 | 71 | 17.45 |
| Strict Jev + Microsoft signal | 712/712 | 714/715 | 1 | 285 | 36.98 |

Exact prices are in `final-results.json`; rounded display values above exclude referral. The final ensemble fixed 15 Jev errors and broke 9 previously correct answers (net +6). The cascade called Microsoft on all 1,000 and Jev on 147. Strict selection called Jev on all 1,000 and Microsoft on 310. All three final response sets cost $0.052557052 in total; a simulated policy pays only for the calls it uses.

The combined 2,000 totals are descriptive sums of working OOF and final: Jev 1918/2000, ensemble 1945/2000, agreement 1852/1865, cascade 1835/1849, strict 1426/1427. They are not a second independent 2,000-item test. Original Jev/Liquid prompts were not rerun on the final thousand, so final outcomes do not independently measure their prompt improvement.

## Evidence map

- `experiments/*/runs.csv`: every exported run, with counts and recovery deduplication.
- `responses.jsonl.gz`: unique raw response records linked to task, corpus hash, prompt template and route.
- `request-templates.json`: exact model/provider/questions, with input state kept separately.
- `corpus-index.json`: IDs, labels, provenance and hashes of full cases.
- `experiments/ag-news/summaries/`: machine-readable experiment comparisons and audits.
- `experiments/proofwriter/working-*`: aligned six prediction sets, OOF decisions, policy results and nested selection results.
- `experiments/proofwriter/final-*`: frozen coefficients, per-item decisions, route breakdown, final/working/combined counts.

Repeated recovery copies of the same response are deduplicated. Separate requests with different response bytes remain separate evidence. Historical baselines reused from the first study are represented in working comparison data; the original first-study corpus/response package remains the upstream source.

Publication metadata replaces personal absolute paths with relative source identities. Where this changes serialized bytes, the published input/frozen hash is recomputed and the original hash is retained under `original_*`. Coefficients, prompts, raw response bytes and numerical outcomes are unchanged. `tools/verify_evidence.py` independently checks all 9,000 working/final model outputs against the packaged raw responses. `tools/summarize_evidence.py` rescores every exported run and reports known and unknown charges, including technical failures. The additional 4,500 upstream ProofWriter baseline responses retain the original five repeats used for stability context.

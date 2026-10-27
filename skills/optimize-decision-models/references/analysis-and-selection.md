# Analyze and choose

Align saved outputs by ID and unchanged gold. Missing calls are missing evidence, not successes. For prompt changes report baseline/candidate correct counts, fixed, broken and categories. Keep unknown costs unknown.

Start with error-count histogram and pairwise overlap. Measure confidence against actual correctness, then correctness conditional on the *predicted* label available at runtime. Low confidence is a risk signal, not proof an answer is wrong. Compare risk-versus-coverage curves rather than relying on a single confidence summary.

Tune thresholds only on development data. For a larger search, use grouped cross-validation with fitting and selection inside each training fold; score outer folds without refitting on their labels. Calibration also belongs inside training folds. The ProofWriter reference implementation illustrates logistic correctness calibration and nested policy selection for three models and three labels. Its frozen coefficients are specific to that experiment. The generic CLI evaluates user-supplied raw-confidence policies; it does not transplant those coefficients to arbitrary tasks.

Candidate mechanisms:
- Single model with threshold: accept above threshold, refer the rest.
- Agreement: all selected answers must agree; disagreement is referred.
- Majority: require more than half of votes; ties/no majority are referred.
- Conditional majority: accept the primary's confident answer, call the others on the remainder.
- Cascade: ask the next model only when the previous model fails its acceptance threshold; refer if none qualifies.
- Veto/signal rule: the secondary's confidence decides whether to retain the primary's answer. It does not replace it. Validate this dependence on errors empirically.

For each compare accepted correct / accepted, coverage, residual errors, referrals, calls and cost. Show uncertainty and support for small high-accuracy subsets. Zero observed errors does not establish a zero-error process. A 99.9% accepted subset and a model answering every task have different denominators.

Calculate full operating cost as model calls plus referral work; measure the referral handler's quality separately. Replay assumes stored independent responses are available for the calls a policy makes. A verifier shown another model's answer requires new requests. Sum of stored elapsed times is a simulation proxy, not measured deployed latency.

Freeze a small set of chosen policies and their artifacts before opening final outcomes. Report the final separately from development; pooled totals mix evaluation regimes. Recommend according to the user's quality/cost/referral constraints, not the largest headline percentage.

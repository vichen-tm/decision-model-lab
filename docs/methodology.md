# Methodology

1. Define the task and measurable objectives: accepted accuracy, referral capacity, latency and cost.
2. Inspect data and labeling logic. Use actual user examples when available; inspect proxy benchmarks before relying on their scores.
3. Reserve independent groups for final validation. Collect baseline on working data for the same candidates and IDs.
4. Diagnose errors and similar correct cases. Test semantic prompt changes incrementally; count fixes and regressions. Broad samples reused for selection become development data.
5. Analyze saved outputs: native confidence, calibrated correctness, predicted-label reliability, error intersections and policy choices.
6. Fit calibration and tune policy parameters inside training folds. Use outer folds for development estimates when selecting many candidates. Freeze the selected configuration.
7. Evaluate once on reserved data. Report final results separately from working results. If you change rules after seeing final results, acquire a new independent final sample.
8. Choose according to the user's constraints; log and audit production decisions, including randomly sampled automatically accepted items.

Accepted accuracy = correct accepted / accepted. Coverage = accepted / all inputs. Referral = 1 − coverage. Referred items are unresolved, not repaired. The report includes Wilson lower bounds as a description of finite-sample uncertainty, not an assurance about future traffic. Costs of downstream verification and deployed latency need their own measurements.

ProofWriter used one question per distinct theory in two 1,000-item groups. Each group had 400 true, 400 false and 200 unknown; known answers spanned depths 0–5. The official source partition was train; independence here means reserved from our tuning, not certified absence from model pretraining.

The included `decision_lab/calibration.py` and `policies.py` preserve the study's three-model/three-label search primitives. The generic CLI supports arbitrary labels but does not automatically fit this specialized calibration. An agent adapting the method must retain predicted-label (never gold-label) features and fit only on working data.

For continuity with the first study, the generic report also gives a descriptive J point. Let G/E be all correct/incorrect baseline answers and g/e those retained above a threshold. J = g / (G + e) chooses a Jaccard-style compromise; ties favor fewer errors then more retained correct answers. U = g/G − e/E measures retained-correct fraction minus retained-error fraction (undefined if G or E is zero). Neither number is accepted accuracy. This point is selected and measured on the same rows; a deployment threshold still needs independent validation.

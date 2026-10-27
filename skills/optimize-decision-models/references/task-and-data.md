# Task and data

Record a short task specification: inputs, labels and their operational meaning, how labels are justified, error costs, volume, acceptable latency, experiment budget, production budget, referral handler and capacity. Separate budget for finding a system from cost per future item.

Check duplicates, missing context, inconsistent labels, class balance, groups and time. Assign `group_id` for documents from the same source/entity/theory. Split at that unit. For future traffic, a time-ordered holdout may be more realistic than random sampling. Keep the final partition inaccessible during diagnosis and tuning.

When labels are disputed, inspect both erroneous and similar correct items. Record original label, proposed interpretation, evidence, adjudication status and reason. Do not remove difficult cases just because models miss them. Report original and adjudicated evaluations separately, retaining a coverage statement for exclusions. For genuinely multi-label tasks define the acceptance set before scoring; top-two accuracy is a different output contract from one-label accuracy.

Search benchmark papers, source repositories and data cards. Compare actual input format, answer semantics, label justification, difficulty, language, age, source overlap, licensing and available sample size. Shortlist only sources actually inspected. A diagnostic run should reveal whether errors support actionable hypotheses. If the problem is missing production inputs, another benchmark does not fix it.

The supplied split utility groups rows, but does not automatically balance class/depth or select a temporal split. Inspect the resulting distributions and choose the split appropriate to the real task. Never rebalance the final set after seeing model errors.

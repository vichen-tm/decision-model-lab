# Data contracts

Runner cases use the exact format in `examples/dataset.jsonl`. A choice question has a named `instructions` string and an ordered `criteria` mapping label → definition. Gold uses the same question ID with `type: choice` and `label`. Gold is never included in a request body. Case IDs must be unique within language. The generic analysis export expects a single choice question and one repeat per case.

An analysis JSONL row:

```json
{"id":"item-1","gold":"billing","models":{"model-a":{"pred":"billing","confidence":0.9,"cost":0.00001},"model-b":{"pred":"access","confidence":0.7,"cost":0.00002}}}
```

All rows must have the same model set and unique IDs. `confidence` is observed native confidence in [0,1], `cost` is actual USD for that call. Missing confidence/cost may be null. No guessed confidence; missing costs remain unknown. Optional `probabilities` and `elapsed_ms` retain richer evidence. A policy requiring confidence fails if that signal is absent. Analysis without confidence still reports accuracy and overlap.

The split command groups by `group_id`, falling back to ID; it uses a seeded shuffled group split. It does not automatically identify related inputs or stratify classes. Define groups and inspect class/depth/time distributions before running.

Error audit: keep original gold, justification, proposed correction, status and affected IDs. Review similar correct cases as controls. Exclusions and adjudicated scores must be named explicitly. Never silently adjust labels to favor a model. AG News findings are about inspected ambiguous boundaries, not proof that every item in the corpus is unusable.

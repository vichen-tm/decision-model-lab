# Quickstart

Install from the repository root as shown in README. Commands below run from that root.

## Prepare a task

`examples/dataset.jsonl` contains two synthetic support classification cases. Each row has `id`, `suite`, `language`, `request` and `gold`. Add `group_id` when examples share an underlying source. See [data contract](data-and-labels.md).

```sh
python -m decision_lab split examples/dataset.jsonl --output output/split --seed 42
python -m decision_lab estimate --requests 200 --input-tokens 300 --output-tokens 20 --input-per-million 0.1 --output-per-million 0.2
python -m decision_lab prepare examples/dataset.jsonl examples/routes.json --run-dir runs/demo --concurrency 1
python -m decision_lab report-run --run-dir runs/demo
```

The route example is the historical Jev route used in the study. Check current availability/version before a live run; a changed provider/version is rejected by the scorer. Replace the route JSON with the exact candidate you intend to test.

## Authorized API run

Supply the key through your shell's secret-management mechanism as `OPENROUTER_API_KEY`; do not put it in a command committed to Git or paste it into chat. `.env.example` documents the variable; the CLI does not automatically load `.env`.

```sh
python -m decision_lab run --run-dir runs/demo --send
python -m decision_lab report-run --run-dir runs/demo
python -m decision_lab export-run --run-dir runs/demo --output output/my-responses.jsonl
python -m decision_lab analyze output/my-responses.jsonl --output output/my-analysis.json
```

Prepare a separate run directory for each prompt/dataset version. For comparison, use the same model name in aligned before/after files:

```sh
python -m decision_lab compare output/baseline.jsonl output/candidate.jsonl --output output/changes.json
```

These last two input files are outputs you collect, not bundled examples. Comparison reports fixed/broken IDs and class counts.

## Choose a policy

Copy `examples/policy.json`, select model IDs in your responses and tune thresholds on development data. Freeze that JSON before applying it to a separate final file. Evaluate with `decision-lab evaluate RESPONSES POLICY --output REPORT`. Add `--referral-cost PRICE_PER_ITEM` to calculate model plus referral expenditure. This adds cost, not assumed accuracy of the referral handler.

See [methodology](methodology.md) for validation and [running experiments](running-experiments.md) for recovery.

## Reproduce the research

```sh
python tools/reproduce.py
python tools/search_working.py --output output/working-search --nested
python -m decision_lab report experiments/proofwriter/final-decisions.jsonl --output output/final-report
```

The original policy search runs on the working thousand only; the second command refits all candidates and verifies the nested-selection outcomes. It writes into a new directory and sends no API requests. The report command produces Markdown, CSV, JSON and an SVG accuracy/coverage plot.

For an exact benchmark rerun, obtain the source archive linked in ATTRIBUTION first. Restore a recorded run into a new directory, then prepare/send using its dataset and routes:

```sh
python tools/restore_inputs.py experiments/proofwriter --run final-validation/jev --source /path/to/proofwriter-dataset-V2020.12.3.zip --output output/restored
python -m decision_lab prepare output/restored/dataset.jsonl output/restored/routes.json --run-dir runs/restored
```

Restoration validates the original archive and every reconstructed case hash. It sends nothing. For AG News provide a JSONL export of the original ordered test split with `text` fields; the tool verifies exact reconstructed hashes and fails on altered/mismatched source data. Recorded run names are listed in `runs.csv`.

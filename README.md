# Decision Model Lab

Give an AI agent a practical workflow for improving finite-choice decisions: inspect your data, compare affordable models, test prompt changes, and choose a system by accuracy, cost and referral work.

**Start with the [agent skill](skills/optimize-decision-models/SKILL.md).** Tell your agent:

> Read AGENTS.md and the optimize-decision-models skill. Help me evaluate my task. Use the context you already have, ask for missing data and budget, and propose a bounded first step before paid calls.

You can also copy the `skills/optimize-decision-models` folder into your agent's skills directory. Keep a clone of this repository available for its scripts and examples. No skill installation is required to read it directly.

## What we found

On a reserved 1,000-task ProofWriter sample, with rules frozen before evaluation:

| System | Correct / accepted | Accepted accuracy | Referred | Model-call cost, USD / million inputs |
|---|---:|---:|---:|---:|
| Jev alone | 955 / 1,000 | 95.50% | 0% | 32.91 |
| Conditional ensemble | 961 / 1,000 | 96.10% | 0% | 35.95 |
| Confidence + agreement | 928 / 937 | 99.04% | 6.3% | 35.95 |
| Microsoft → Jev cascade | 923 / 929 | 99.35% | 7.1% | 17.45 |
| Strict selection | 714 / 715 | 99.86% | 28.5% | 36.98 |

Prices are measured historical model-call costs, excluding downstream referral. Accepted accuracy describes the subset the policy answers automatically. [Full findings and experimental history](docs/experiment-results.md).

## Run offline first

Python 3.10+ on macOS/Linux (the durable runner uses POSIX file locks):

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m decision_lab analyze examples/responses.jsonl --output output/example-analysis.json
python -m decision_lab evaluate examples/responses.jsonl examples/policy.json --output output/example-policy.json
python tools/reproduce.py
python -m unittest discover -s tests
```

The 12-row example demonstrates the interface; it is not a validation sample. `reproduce.py` checks every final policy decision, counts and costs, working out-of-fold counts, and combined totals against the recorded study.

## Use your own data

Follow [quickstart](docs/quickstart.md). The bundled runner prepares immutable plans and sends through **OpenRouter Decisions**, with incremental raw-response journaling, pinned routes, bounded concurrency and explicit recovery. Keys are read from `OPENROUTER_API_KEY`. Preparing a plan sends nothing; sending requires `run --send`.

The skill can research candidates and guide new hypotheses. The generic analysis CLI supports arbitrary finite labels and raw-confidence policies. Study-specific calibrated policies and the original search implementation are included separately for reproducibility.

## Where things live

- `skills/`: the agent workflow and focused methodology references.
- `decision_lab/`: analysis, comparisons, policy replay and CLI.
- `decision_benchmark/`: reused request/scoring/journal/runner core from the first study.
- `experiments/`: run inventories, response evidence, prompt templates, corpus identities and study results.
- `docs/`: data contract, experiment history, interpretation and operating instructions.
- `tools/`: evidence packaging and reproducibility checks.
- `examples/`: runnable offline sample and a small custom task.
- `releases/manifest.json`: publication inventory and checksums.

### Data distribution

The public evidence contains model responses, labels, IDs, prompts, routes, costs and input hashes. Third-party benchmark **input text is excluded** from Git while redistribution terms remain unresolved. A local private corpus archive preserves the original cases; source URLs and identities are in [ATTRIBUTION.md](ATTRIBUTION.md). Offline result analysis requires no benchmark download. New benchmark inference requires obtaining the source data under its terms and reconstructing the inputs; your own labeled JSONL works directly.

Related first study: [ten-model benchmark](https://github.com/vichen-tm/decisions-model-banchmark).

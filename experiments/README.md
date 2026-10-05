# Running agents in the source-failure environment

All commands run from this directory.

## Setup

```bash
pip install -r ../requirements.txt
```

* **Open models** run locally with vLLM. Put the official weights under `models/`, in the directories listed in
  `config.py` (`MODELS`): `Qwen2.5-7B-Instruct`, `Llama-3.1-8B-Instruct`, `Qwen3-8B`, `Qwen3-32B`. To add a model,
  add a line to `MODELS`. Use `--gpu 0,1,2,3` for tensor parallelism over several GPUs.
* **Claude models** run through the official Anthropic API (`ANTHROPIC_API_KEY`); their keys in `MODELS` start with
  `anthropic:`.
* **HotpotQA** (distractor development set) is downloaded on first use. The question lists are in `manifests/`:
  `dev100` for development, `test300` for the main results, `fresh300` for a fresh replication.

## Run

```bash
python run_gate2.py --arm none --model qwen3-8b --split test300 --gpu 0
```

runs one condition for one model in every failure regime and writes a run directory under `results_v2/` with one
JSON file of trajectories per regime.

| condition | `--arm` |
|---|---|
| unaided | `none` |
| permit (may answer from memory) | `prompt_permit` |
| budget (eight actions, counter) | `prompt_budget` |
| stopping rule stated in the prompt | `prompt_policy` |
| price per call stated | `prompt_cost` |
| decide (running count shown, agent decides) | `own_reason` |
| enforced rule (answer after 5 consecutive useless judgments) | `rule_k5_side` (other thresholds: `rule_k3_side`, ...) |
| budget + enforced rule | `combo_budget_rule_k5` |
| enforced rule on the judgments stated in the agent's reasoning | `rule_k5_stated` |
| enforced rule driven by a random or a lexical signal (controls) | `rule_k5_side_randmatch`, `rule_k5_side_lex` |
| reasoning mode | add `--thinking` |

| script | what it runs |
|---|---|
| `run_gate2.py` | HotpotQA: clean, persistent, recover_after_1/2/3, late_onset_from_3, plausible |
| `run_gate2_fever.py` | FEVER fact verification (`fever_data.json`) |
| `run_gate2_ext.py` | answerless source, longer recovery (`recover_after_4/5`), backup tool (`ts_*` regimes) |
| `run_judge_replay.py --run-dir <run dir>` | the agent's judgment of every observation, replayed on a finished run |

For many jobs over several GPUs, write one job per line as `script|model|arm|split|tag||extra args` and run
`./job_pool.sh jobs.txt results_v2/index.log 0 1 2 3`.

## Evaluate

```bash
python evaluate.py results_v2/<run dir>                                   # enforced rule, combination, decide
python run_judge_replay.py --run-dir results_v2/<run dir> --gpu 0         # other conditions: replay judgments first
python evaluate.py results_v2/<run dir> --judgments results_v2/<replay dir>
```

prints success and tool calls per regime, mean6 (equal weight over the six regimes other than plausible), the
time-matched contrast Δ with a bootstrap interval, and how often the agent answers after five consecutive useless
judgments. For the unaided Qwen3-8B on `test300` it prints mean6 0.448 and Δ = −0.065 [−0.095, −0.039]; with
`rule_k5_side`, 0.474 and Δ = +0.329 [+0.286, +0.373].

The prompt text of every condition is in [`../prompts/`](../prompts).

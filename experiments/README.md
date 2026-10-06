# Running agents in the source-failure environment

All commands run from this directory.

| | |
|---|---|
| `run_gate2.py` | HotpotQA: clean, persistent, recover_after_1/2/3, late_onset_from_3, plausible |
| `run_gate2_fever.py` | FEVER fact verification |
| `run_gate2_ext.py` | answerless source, longer recovery (`recover_after_4/5`), backup tool (`ts_*` regimes) |
| `run_judge_replay.py` | the agent's judgment of every observation, replayed on a finished run |
| `evaluate.py` | success per regime, mean6, Δ and the answer rate after five useless judgments for one run |
| `reproduce_paper.py` | every Δ in Figure 3 from the released episodes |
| `episodes/` | every episode of the paper's main experiments ([format](episodes/README.md)) |
| `prompts/` | every prompt used in the paper, as plain text |
| `manifests/` | the question lists (`dev100`, `test300`, `fresh300`) and the FEVER claims |
| `core/` | the agent, environment and model backends used by the scripts |

## Setup

```bash
pip install -e "..[experiments]"
```

* **Open models** run locally with vLLM. Put the official weights under `models/`, in the directories listed in
  `core/config.py` (`MODELS`): `Qwen2.5-7B-Instruct`, `Llama-3.1-8B-Instruct`, `Qwen3-8B`, `Qwen3-32B`. To add a
  model, add a line to `MODELS`. Use `--gpu 0,1,2,3` for tensor parallelism over several GPUs.
* **Claude models** run through the official Anthropic API (`ANTHROPIC_API_KEY`); their keys in `MODELS` start with
  `anthropic:`.
* **HotpotQA** (distractor development set) is downloaded on first use.

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

## Evaluate

```bash
python evaluate.py results_v2/<run dir>                                   # enforced rule, combination, decide
python run_judge_replay.py --run-dir results_v2/<run dir> --gpu 0         # other conditions: replay judgments first
python evaluate.py results_v2/<run dir> --judgments results_v2/<replay dir>
```

For the unaided Qwen3-8B on `test300` it prints mean6 0.448 and Δ = −0.065 [−0.095, −0.039]; with `rule_k5_side`,
0.474 and Δ = +0.329 [+0.286, +0.373].

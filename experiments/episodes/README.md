# Episodes

The behaviour of every model in every condition of the paper's main experiments, one record per episode, in the
toolkit's format. 43 files: four open models × eight conditions on the 300 test questions (`test300/`), and the
unaided, budget and enforced-rule conditions on 300 fresh questions (`fresh300/`).

```python
from judged_useless import load_jsonl, time_matched_contrast

FAIL = {"persistent", "recover_after_1", "recover_after_2", "recover_after_3", "late_onset_from_3"}
eps = load_jsonl("experiments/episodes/test300/qwen3-8b__unaided.jsonl.gz")
time_matched_contrast([e for e in eps if e["regime"] in FAIL])   # Δ = -0.065 [-0.095, -0.039]
```

`python experiments/reproduce_paper.py` recomputes every Δ in Figure 3 and the open models' mean6 in Table 2 from these files and
checks them against the paper (all 43 cells match).

## Fields

| field | meaning |
|---|---|
| `model` | `qwen2.5-7b` (Qwen2.5-7B-Instruct), `llama3.1-8b` (Llama-3.1-8B-Instruct), `qwen3-8b` (Qwen3-8B), `qwen3-32b` (Qwen3-32B); the Qwen3 models in non-thinking mode |
| `condition` | `unaided`, `permit`, `budget`, `stated_rule`, `call_cost`, `decide`, `enforced_rule`, `combo` (Table 1 of the paper) |
| `split` | `test300` or `fresh300` (question lists in `../manifests/`) |
| `question_id` | HotpotQA question id (distractor development set) |
| `regime` | which observations fail: `clean`, `persistent`, `recover_after_1/2/3`, `late_onset_from_3`, `plausible` |
| `actions` | the action at each step: `search`, `lookup`, `finish`, or `invalid` for an action the environment does not support (it replies with an error message, and the step counts toward the budget of eight) |
| `judgments` | the agent's one-word judgment of the observation returned by each action other than `finish`: `USELESS`, `USEFUL`, `UNPARSED` (a reply with neither word, 4 cases), or null (see below) |
| `success` | whether the final answer is correct (token F1 ≥ 0.6) |
| `termination` | `answered` (the last action is `finish`), `budget_exhausted` (eight actions without `finish`), `format_failure` (a turn with no parsable action) or `generation_truncated` (a turn cut off at the token limit); the last two end the episode without an answer, in 41 episodes at the first turn, so their `actions` are empty |

Judgments come from the side-channel question: recorded during the run in the conditions that use them (decide,
enforced rule, combo) and replayed on the recorded trajectories otherwise. A judgment is null in two cases:

* in the replayed conditions, judgments were replayed for the five failure regimes only, so `clean` and `plausible`
  episodes have none;
* in the recorded conditions, when the budget runs out, the observation returned by the eighth action precedes no
  further decision and is not judged.

## Notes

* Δ in the paper pools the five failure regimes in `FAIL` above; the clean and plausible regimes are not used for Δ.
* mean6 is the success averaged with equal weight over `clean` and the five failure regimes.
* The decide condition was run for the three 7–8B models only.

License: CC BY 4.0.

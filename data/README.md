# Episodes

The behaviour of every model in every condition of the paper's main experiments, one record per episode, in the
toolkit's format. 43 files: four open models × eight conditions on the 300 test questions (`test300/`), and the
unaided, budget and enforced-rule conditions on 300 fresh questions (`fresh300/`).

```python
from judged_useless import load_jsonl, time_matched_contrast

FAIL = {"persistent", "recover_after_1", "recover_after_2", "recover_after_3", "late_onset_from_3"}
eps = load_jsonl("data/episodes/test300/qwen3-8b__unaided.jsonl.gz")
time_matched_contrast([e for e in eps if e["regime"] in FAIL])   # Δ = -0.065 [-0.095, -0.039]
```

`python data/reproduce_paper.py` recomputes every Δ in Figure 3 and every mean6 in Table 2 from these files and
checks them against the paper (all 43 cells match).

## Fields

| field | meaning |
|---|---|
| `model` | `qwen2.5-7b`, `llama3.1-8b`, `qwen3-8b`, `qwen3-32b` |
| `condition` | `unaided`, `permit`, `budget`, `stated_rule`, `call_cost`, `decide`, `enforced_rule`, `combo` (Table 1 of the paper) |
| `split` | `test300` or `fresh300` (question lists in `../experiments/manifests/`) |
| `question_id` | HotpotQA question id |
| `regime` | `clean`, `persistent`, `recover_after_1/2/3`, `late_onset_from_3`, `plausible` |
| `actions` | the action at each step: `search`, `lookup`, `finish`, or `invalid` when the agent's turn contained no valid action (it counts as not answering) |
| `judgments` | the agent's judgment of each observation (`judgments[k]` judges the observation returned by `actions[k]`): `USELESS`, `USEFUL`, `UNPARSED` (a reply with neither word, 4 cases), or `null` where no judgment was elicited, e.g. after the final action or in regimes where judgments were not replayed |
| `success` | whether the final answer is correct (token F1 ≥ 0.6) |

Judgments are the agent's side-channel judgments: recorded during the run in the conditions that use them (decide,
enforced rule, combo) and replayed on the recorded trajectories otherwise.

## Notes

* Δ in the paper pools the five failure regimes in `FAIL` above; the clean and plausible regimes are not used for Δ.
* mean6 is the success averaged with equal weight over `clean` and the five failure regimes.
* The decide condition was run for the three 7–8B models only.

The full trajectories (with the agents' text) and the annotation labels will be added later.

License: CC BY 4.0.

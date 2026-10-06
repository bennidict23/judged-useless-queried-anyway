# Using the toolkit

## Measure Δ

Log each episode as one JSON line with the agent's actions and its judgment of each observation:

```json
{"question_id": "q17", "actions": ["search", "search", "lookup", "search", "finish"],
 "judgments": ["USELESS", "USEFUL", "USELESS", "USELESS"]}
```

The k-th judgment is about the observation returned by the k-th action, and is USELESS, USEFUL or null. Get it by
asking a one-word question on a copy of the conversation
([experiments/prompts/08_side_channel_judgment_question.txt](../experiments/prompts/08_side_channel_judgment_question.txt)), or read it from
the agent's own reasoning with `stated_judgment()`.

```python
from judged_useless import load_jsonl, time_matched_contrast, answer_rate_after_run

episodes = load_jsonl("my_episodes.jsonl")   # or a file from experiments/episodes/
time_matched_contrast(episodes)        # {'delta': ..., 'ci_low': ..., 'ci_high': ..., ...}
answer_rate_after_run(episodes, k=5)   # how often it answers after 5 useless judgments in a row
```

Δ pools the decisions after 3 to 6 observations and leaves out the final action of the eight-action budget, which is
the last chance to answer; with a budget of *B* actions, pass `t_max = B - 2`. It needs both kinds of histories at the
same step, so collect episodes in which a source fails from the start as well as episodes in which it fails later or
recovers. Cells are pooled with Mantel–Haenszel weights, and the interval comes from a bootstrap over questions.

## Enforce the integration step

```python
from judged_useless import IntegrationRule, stated_judgment

rule = IntegrationRule(k=5, source="stated")
if rule.update(stated_judgment(thought)):   # after each observation, on the agent's latest thought
    ...                                     # allow only the answer action from now on
```

With `source="stated"`, a USEFUL judgment resets the run and a thought without an explicit judgment leaves it
unchanged. With `source="side_channel"`, pass the parsed reply to the side-channel question; anything other than
USELESS resets the run. When the rule fires, `FORCE_SYSTEM_SUFFIX` and `FORCE_USER_MESSAGE` are the prompts we used
to switch the agent to answering, and `python run_gate2.py --arm rule_k5_side` in [experiments](../experiments) runs
the rule in the full environment.

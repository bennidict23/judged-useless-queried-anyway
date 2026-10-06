"""Prompt-only control arms for Gate 2 (addendum 3; see the pre-registration).

These arms test the alternative explanation that agents keep searching only because (a) the task prompt never
says that answering from one's own knowledge is allowed, or (b) the agent does not know its action budget.
They change nothing but text the agent sees; there is no judgment call, no decision step and no forced finish.

  prompt_permit  system prompt + PERMIT_SUFFIX
  prompt_budget  system prompt + PERMIT_SUFFIX + BUDGET_SUFFIX, and every observation (and the question) is
                 followed by a remaining-actions counter.
  prompt_policy  system prompt + PERMIT_SUFFIX + POLICY_SUFFIX: the k=5 stopping rule stated in words (addendum 11);
                 the agent must count and apply it itself.
  prompt_cost    prompt_budget + COST_SUFFIX: the scoring, including a per-call cost, is stated (addendum 12b).

Installed as run_pilot.PRE_ACTION_HOOK; called before every action generation with the undone runs.
"""
from __future__ import annotations

from config import MAX_AGENT_STEPS

PROMPT_ARMS = ('prompt_permit', 'prompt_budget', 'prompt_policy', 'prompt_cost')

PERMIT_SUFFIX = ("\n\nIf the search results do not give you the information you need, you may also answer from your "
                 "own knowledge: use finish[answer] with your best guess.")
BUDGET_SUFFIX = (f"\nYou can take at most {MAX_AGENT_STEPS} actions for each question, including finish. If you have "
                 f"not used finish[answer] by then, the question is scored as wrong.")


POLICY_SUFFIX = ("\nStopping policy: if five consecutive search or lookup results in a row are useless for answering the "
                 "question, stop consulting the knowledge base and answer from your own knowledge with finish[answer].")

COST_SUFFIX = ("\nScoring: a correct final answer earns 1 point; each search or lookup costs 0.05 points; a wrong or missing "
               "answer earns 0. Maximize your total points.")


def counter_text(remaining: int) -> str:
    return f"\n\n[Actions remaining: {remaining}]"


class PromptArm:
    def __init__(self, arm: str):
        if arm not in PROMPT_ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.budget = arm in ('prompt_budget', 'prompt_cost')
        self.cost = arm == 'prompt_cost'
        self.policy = arm == 'prompt_policy'
        self.log = {}  # kept for interface parity with DecisionStep (run_gate2 reads hook.log)

    def __call__(self, llm, runs):
        for r in runs:
            msgs = r['messages']
            if not r.get('_prompt_arm_applied'):
                msgs[0] = {'role': msgs[0]['role'], 'content': msgs[0]['content'] + PERMIT_SUFFIX + (BUDGET_SUFFIX if self.budget else '')
                           + (POLICY_SUFFIX if self.policy else '') + (COST_SUFFIX if self.cost else '')}
                r['_prompt_arm_applied'] = True
            if self.budget:
                # annotate the newest user turn exactly once: the question at step 0, the latest observation afterwards.
                # State lives on the run dict, never inside messages (chat templates / APIs only accept role+content).
                i = len(msgs) - 1
                if msgs[i]['role'] == 'user' and r.get('_counter_idx') != i:
                    n_done = len(r['trajectory'].steps)
                    msgs[i] = {'role': 'user', 'content': msgs[i]['content'] + counter_text(MAX_AGENT_STEPS - n_done)}
                    r['_counter_idx'] = i


# Combination arm (addendum 6): the budget prompt and the side-channel rule together.
COMBO_ARMS = tuple(f'combo_budget_rule_k{k}' for k in range(2, 8))


class Combo:
    """prompt_budget text (system suffix + remaining-actions counter) and the rule_k{K}_side stop rule, applied in
    that order before every action. Neither component is modified; the rule's judgments are logged as usual."""

    def __init__(self, arm: str):
        if arm not in COMBO_ARMS:
            raise ValueError(arm)
        from gate2_intervention import DecisionStep
        k = int(arm.rsplit('_k', 1)[1])
        self.prompt = PromptArm('prompt_budget')
        self.rule = DecisionStep(f'rule_k{k}_side')
        self.log = self.rule.log

    def __call__(self, llm, runs):
        self.prompt(llm, runs)
        self.rule(llm, runs)

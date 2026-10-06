"""Amendment 19: the enforced rule driven by the judgment the agent states in its own reasoning (no judgment call).

After every observation the agent writes its usual Thought/Action turn. Before the action is executed, the harness
reads the Thought with the zero-cost keyword reader (stated_parser.stated_judgment) and updates the run of consecutive
stated-useless judgments: USELESS adds one, USEFUL resets it, NONE (no explicit judgment, including an empty Thought)
leaves it unchanged. When the run reaches K and the agent's proposed action is search or lookup, that turn is discarded
and the agent is asked again with exactly the enforced rule's finish instruction (STRICT system suffix and
F_PROTOCOL_MAIN after the observation, as in rule_k5_side); from then on only finish remains. The timing matches the
side-channel rule: the judgment of observation t is read before action t+1 is executed.

Arms: rule_k5_stated (base prompt) and combo_budget_rule_k5_stated (budget prompt + the same rule).
Installed as run_pilot.POST_GENERATION_HOOK (the budget prompt, if any, as PRE_ACTION_HOOK).
"""
from __future__ import annotations

from agent import parse_agent_response
from config import F_PROTOCOL_MAIN, F_PROTOCOL_STRICT_SYSTEM_SUFFIX
from stated_parser import stated_judgment

STATED_ARMS = ('rule_k5_stated', 'combo_budget_rule_k5_stated')


class StatedRule:
    def __init__(self, arm: str, k: int = 5):
        if arm not in STATED_ARMS:
            raise ValueError(arm)
        self.arm, self.k = arm, k
        self.budget = arm.startswith('combo')
        self.log = {}           # (question_id, feedback_mode) -> per-decision records
        self.regenerated = 0    # discarded turns (one per firing): counted as extra generations

    def pre(self):
        if not self.budget:
            return None
        from prompt_arms import PromptArm
        return PromptArm('prompt_budget')

    @staticmethod
    def _force_finish(run):
        msgs = run['messages']
        if not msgs[0]['content'].endswith(F_PROTOCOL_STRICT_SYSTEM_SUFFIX):
            msgs[0] = {'role': 'system', 'content': msgs[0]['content'] + F_PROTOCOL_STRICT_SYSTEM_SUFFIX}
        if not msgs[-1]['content'].endswith(F_PROTOCOL_MAIN):
            msgs[-1] = {'role': 'user', 'content': msgs[-1]['content'] + '\n\n' + F_PROTOCOL_MAIN}

    def __call__(self, llm, undone, results, metadata):
        fire = []
        for idx, (run, (response, _)) in enumerate(zip(undone, results)):
            tr = run['trajectory']
            if not tr.steps:            # first action: no observation to judge yet
                continue
            st = run.setdefault('stated_rule', {'k': 0, 'fired': False})
            thought, action, _ = parse_agent_response(response, strict=getattr(tr, 'shuffle_mode', '') == 'per_question')
            j = stated_judgment(thought)
            if j == 'USELESS':
                st['k'] += 1
            elif j == 'USEFUL':
                st['k'] = 0
            fires = (st['fired'] or st['k'] >= self.k) and action in ('search', 'lookup')
            self.log.setdefault((tr.question_id, tr.feedback_mode), []).append({
                'question_id': tr.question_id, 'feedback_mode': tr.feedback_mode, 't': len(tr.steps),
                'stated_judgment': j, 'k_useless_stated': st['k'], 'proposed_action': action,
                'decision': 'ANSWER_NOW' if fires else 'CONTINUE', 'own_judgment': j,
                'last_obs_corrupted': bool(run['env'].observation_meta[-1].get('corrupted')) if run['env'].observation_meta else None})
            if fires:
                st['fired'] = True
                fire.append(idx)
        if not fire:
            return results, metadata
        for idx in fire:
            self._force_finish(undone[idx])
        outs = llm.batch_chat([undone[i]['messages'] for i in fire], max_tokens=getattr(llm, 'max_tokens', None))
        meta2 = getattr(llm, 'last_batch_metadata', [{} for _ in outs])
        results, metadata = list(results), list(metadata)
        for i, out, m2 in zip(fire, outs, meta2):
            metadata[i] = dict(m2, discarded_response=results[i][0], discarded_tokens=results[i][1])
            results[i] = out
            self.regenerated += 1
        return results, metadata

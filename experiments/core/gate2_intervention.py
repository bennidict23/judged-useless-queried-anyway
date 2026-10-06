"""Gate 2 pilot intervention: a lightweight pre-action decision step.

Before each action (after at least one observation) the agent is asked, in two short turns:
  stage 1  JUDGMENT: was the last observation useful?  (one word)
  stage 2  DECISION: CONTINUE or ANSWER_NOW, given that judgment and the streak of useless judgments.
Then the normal Thought/Action turn follows with an instruction to act consistently with the decision.

Arms (what is shown as the judgment in stage 2):
  own         the agent's own stage-1 answer                      (the intervention)
  random      a coin flip, replacing the agent's answer            (signal ablation: same structure, no signal)
  design      the environment's corruption flag                    (oracle-ish upper reference)
  decide_only no stage 1; stage 2 asks for a decision with no judgment mentioned (structure-only ablation)
The agent's own stage-1 answer is always elicited and logged, even when it is not shown.
"""
from __future__ import annotations
import random, re
from config import SEED, stable_seed, F_PROTOCOL_MAIN, F_PROTOCOL_STRICT_SYSTEM_SUFFIX
from lexical_detector import lexical_useless

JUDGE_PROMPT = ("Before acting: was the LAST observation useful for answering the original question? "
                "Reply with exactly one word: USEFUL or USELESS.")
DECIDE_PROMPT_J = ("Your judgment of the last observation: {j}. Consecutive observations judged USELESS so far: {k}. "
                   "Decide now. Reply with exactly one word: CONTINUE (consult the knowledge base again) or "
                   "ANSWER_NOW (submit your best answer from what you already know).")
DECIDE_PROMPT_NOJ = ("Decide now. Reply with exactly one word: CONTINUE (consult the knowledge base again) or "
                     "ANSWER_NOW (submit your best answer from what you already know).")
ACT_PROMPT = ("Now continue in the required format (Thought / Action / Action Input), acting consistently with your "
              "decision: CONTINUE means a search or lookup action; ANSWER_NOW means the finish action with your best short "
              "answer to the original question as the Action Input.")
REASON_PROMPT_J = ("Your judgment of the last observation: {j}. Consecutive observations judged USELESS so far: {k}. "
                   "In one or two sentences, weigh how likely you are to succeed if you keep consulting this knowledge base "
                   "against how likely you are to answer correctly right now from what you already know. "
                   "Then, on a final line, write exactly: DECISION: CONTINUE or DECISION: ANSWER_NOW.")
REASON_PROMPT_NOJ = ("In one or two sentences, weigh how likely you are to succeed if you keep consulting this knowledge base "
                     "against how likely you are to answer correctly right now from what you already know. "
                     "Then, on a final line, write exactly: DECISION: CONTINUE or DECISION: ANSWER_NOW.")
CAL_SENTENCE = (" Note: this knowledge base sometimes returns one or two useless results and then recovers, so stopping "
                "before three consecutive useless observations is usually premature; after three or more, continuing rarely helps.")
REASON_PROMPT_CAL = REASON_PROMPT_J.replace("Consecutive observations judged USELESS so far: {k}. ",
                                            "Consecutive observations judged USELESS so far: {k}." + CAL_SENTENCE + " ")
ARMS = {'own', 'random', 'design', 'decide_only', 'own_reason', 'random_reason', 'design_reason', 'decide_only_reason',
        'own_reason_cal', 'rule_k2', 'rule_k3', 'rule_k4', 'rule_k3_random', 'rule_k3_design', 'rule_k3_hard', 'rule_k4_hard', 'rule_k2_side', 'rule_k3_side', 'rule_k4_side', 'rule_k5_side', 'rule_k6_side', 'rule_k3_side_random', 'rule_k4_side_random', 'rule_k5_side_random'}
ARMS |= {f'rule_k{k}_side{s}' for k in range(2, 8) for s in ('', '_random', '_lex', '_design', '_randmatch')}


def parse_judgment(text):
    t = text.upper()
    if 'USELESS' in t:
        return 'USELESS'
    if 'USEFUL' in t:
        return 'USEFUL'
    return 'UNPARSED'


def parse_decision(text):
    t = text.upper().replace(' ', '_')
    m = re.findall(r'DECISION:\s*_*(ANSWER_NOW|CONTINUE)', t)
    if m:
        return m[-1]
    ia, ic = t.rfind('ANSWER_NOW'), t.rfind('CONTINUE')
    if ia < 0 and ic < 0:
        return 'UNPARSED'
    return 'ANSWER_NOW' if ia > ic else 'CONTINUE'


class DecisionStep:
    def __init__(self, arm: str):
        assert arm in ARMS
        self.arm = arm
        self.rule_k = int(arm[6:].split('_')[0]) if arm.startswith('rule_k') else None
        self.hard = arm.endswith('_hard')   # at trigger, force the finish action with the STRICT F wording
        # side-channel: judgments elicited on a scratch copy; the acting history is untouched except a forced
        # STRICT finish at trigger. Purest test of 'a rule on the agent's own judgment', zero interference otherwise.
        self.side = '_side' in arm
        self.cal = arm == 'own_reason_cal'
        self.reason = arm.endswith('_reason') or self.cal
        if self.rule_k:
            self.base = ('random' if arm.endswith('_random') else 'design' if arm.endswith('_design') else
                         'lex' if arm.endswith('_lex') else 'randmatch' if arm.endswith('_randmatch') else 'own')
            if self.base == 'randmatch':
                # rate-matched random signal: P(USELESS) = the model's own useless-judgment rate on dev (equal regime weights);
                # isolates whether the judgments' ALIGNMENT with actual uselessness matters, not just how often they say useless
                import os
                if 'RANDMATCH_P' not in os.environ:
                    raise RuntimeError('RANDMATCH_P must be set for _randmatch arms')
                self.p_useless = float(os.environ['RANDMATCH_P'])
            if self.side:
                self.hard = True
        else:
            self.base = 'own' if self.cal else arm.replace('_reason', '')
        self.log = {}  # (question_id, feedback_mode) -> list of step records

    def _state(self, run):
        return run.setdefault('gate2', {'k_useless': 0})

    def __call__(self, llm, undone):
        runs = [r for r in undone if r['trajectory'].steps]
        if not runs:
            return
        shown = {}
        own = {}
        if self.base != 'decide_only':
            outs = llm.batch_chat([r['messages'] + [{'role': 'user', 'content': JUDGE_PROMPT}] for r in runs],
                                  max_tokens=8, stop=[])
            for r, (text, _) in zip(runs, outs):
                own[id(r)] = (text, parse_judgment(text))
                tr = r['trajectory']; t = len(tr.steps)
                if self.base == 'own':
                    # An unparsed reply is not an explicit 'useless' judgment: it does not count toward the streak
                    # (the streak resets). The rule acts only on explicit USELESS. (Amendment 2: Haiku
                    # sometimes ignores the one-word format and starts a positive Thought; local runs had 0 unparsed.)
                    j = own[id(r)][1]
                elif self.base == 'random':
                    j = random.Random(stable_seed(SEED, 'gate2-random', tr.question_id, tr.feedback_mode, t)).choice(['USEFUL', 'USELESS'])
                elif self.base == 'randmatch':
                    u = random.Random(stable_seed(SEED, 'gate2-randmatch', tr.question_id, tr.feedback_mode, t)).random()
                    j = 'USELESS' if u < self.p_useless else 'USEFUL'
                elif self.base == 'lex':
                    j = 'USELESS' if lexical_useless(tr.question, r['messages'][-1]['content']) else 'USEFUL'
                else:
                    meta = r['env'].observation_meta
                    j = 'USELESS' if (meta and meta[-1].get('corrupted')) else 'USEFUL'
                shown[id(r)] = j
                st = self._state(r)
                st['k_useless'] = st['k_useless'] + 1 if j == 'USELESS' else 0
                if not self.side:
                    r['messages'] += [{'role': 'user', 'content': JUDGE_PROMPT}, {'role': 'assistant', 'content': j}]
        if self.rule_k:
            # fixed-threshold rule: no model decision; ANSWER_NOW iff the (own-judged) useless streak >= K
            for r in runs:
                k = self._state(r)['k_useless']; d = 'ANSWER_NOW' if k >= self.rule_k else 'CONTINUE'
                dp = DECIDE_PROMPT_J.format(j=shown[id(r)], k=k)
                if self.hard and d == 'ANSWER_NOW':
                    if not r['messages'][0]['content'].endswith(F_PROTOCOL_STRICT_SYSTEM_SUFFIX):
                        r['messages'][0] = {'role': 'system', 'content': r['messages'][0]['content'] + F_PROTOCOL_STRICT_SYSTEM_SUFFIX}
                    if self.side:
                        # the last message is the observation; append the finish instruction as the same user turn
                        r['messages'][-1] = {'role': 'user', 'content': r['messages'][-1]['content'] + '\n\n' + F_PROTOCOL_MAIN}
                    else:
                        r['messages'] += [{'role': 'user', 'content': dp}, {'role': 'assistant', 'content': d}, {'role': 'user', 'content': F_PROTOCOL_MAIN}]
                elif self.side:
                    pass  # CONTINUE: history untouched
                else:
                    r['messages'] += [{'role': 'user', 'content': dp}, {'role': 'assistant', 'content': d}, {'role': 'user', 'content': ACT_PROMPT}]
                tr = r['trajectory']
                self.log.setdefault((tr.question_id, tr.feedback_mode), []).append({
                    'question_id': tr.question_id, 'feedback_mode': tr.feedback_mode, 't': len(tr.steps),
                    'own_judgment_raw': own[id(r)][0], 'own_judgment': own[id(r)][1], 'shown_judgment': shown[id(r)],
                    'k_useless_shown': k, 'decision_raw': f'RULE_K{self.rule_k}', 'decision': d,
                    'last_obs_corrupted': bool(r['env'].observation_meta[-1].get('corrupted')) if r['env'].observation_meta else None})
            return
        prompts = []
        for r in runs:
            if self.base == 'decide_only':
                dp = REASON_PROMPT_NOJ if self.reason else DECIDE_PROMPT_NOJ
            else:
                dp = (REASON_PROMPT_CAL if self.cal else (REASON_PROMPT_J if self.reason else DECIDE_PROMPT_J)).format(j=shown[id(r)], k=self._state(r)['k_useless'])
            r['_gate2_dp'] = dp
            prompts.append(r['messages'] + [{'role': 'user', 'content': dp}])
        outs = llm.batch_chat(prompts, max_tokens=(160 if self.reason else 8), stop=[])
        for r, (text, _) in zip(runs, outs):
            d = parse_decision(text)
            r['messages'] += [{'role': 'user', 'content': r.pop('_gate2_dp')}, {'role': 'assistant', 'content': text.strip() or d},
                              {'role': 'user', 'content': ACT_PROMPT}]
            tr = r['trajectory']
            key = (tr.question_id, tr.feedback_mode)
            self.log.setdefault(key, []).append({
                'question_id': tr.question_id, 'feedback_mode': tr.feedback_mode, 't': len(tr.steps),
                'own_judgment_raw': own.get(id(r), ('', None))[0], 'own_judgment': own.get(id(r), ('', None))[1],
                'shown_judgment': shown.get(id(r)), 'k_useless_shown': self._state(r)['k_useless'] if self.base != 'decide_only' else None,
                'decision_raw': text, 'decision': d,
                'last_obs_corrupted': bool(r['env'].observation_meta[-1].get('corrupted')) if r['env'].observation_meta else None,
            })

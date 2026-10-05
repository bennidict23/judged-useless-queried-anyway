"""Amendment 13 extensions (the pre-registration).

answerless source  The question's knowledge base with every supporting-fact sentence removed: searches still return
                   the question's own, on-topic pages, but no page contains the facts that answer the question.
two-source setting A second tool, backup_search, searches an intact copy of the same knowledge base and never fails;
                   the prompt calls it slower and more expensive, so the decision is when to abandon the primary
                   search. Failure regimes apply to the primary search only.
RuleSwitch         The rule's analogue for this setting: after k consecutive side-channel USELESS judgments on primary
                   results, the primary search is disabled and the agent is told to use backup_search or finish.
"""
from __future__ import annotations

from tools import HotpotQAEnvironment
from gate2_intervention import JUDGE_PROMPT, parse_judgment

BACKUP_LINE = ("- backup_search[query]: Search a second, independent copy of the knowledge base. It is slower and more "
               "expensive than search: prefer search, and use backup_search when search does not give you useful results.")
SWITCH_NOTICE = "The search tool is no longer available. Use backup_search[query] or finish[answer]."


def answerless_kb(kb: dict[str, str], raw_item: dict) -> dict[str, str]:
    """kb with every supporting-fact sentence of raw_item removed; titles and all other sentences are kept."""
    sf: dict[str, set[int]] = {}
    for title, idx in raw_item['supporting_facts']:
        sf.setdefault(title, set()).add(idx)
    out = {}
    for title, sents in raw_item['context']:
        keep = [s for i, s in enumerate(sents) if i not in sf.get(title, set())]
        out[title] = ' '.join(keep).strip() or '(no further text on this page)'
    if set(out) != set(kb):
        raise ValueError('raw item does not match the knowledge base')
    return out


def system_prompt_with_backup(base: str) -> str:
    """Insert the backup tool after the lookup line of the agent's system prompt."""
    anchor = '- finish[answer]:'
    if anchor not in base:
        raise ValueError('unexpected system prompt')
    return base.replace(anchor, BACKUP_LINE + '\n' + anchor, 1)


class ExtEnvironment(HotpotQAEnvironment):
    def __init__(self, *args, full_kb: dict[str, str] | None = None, backup: bool = False,
                 all_failed: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.backup = backup
        self.all_failed = all_failed          # design label: every primary observation lacks the facts (answerless)
        self.primary_disabled = False
        self._backup_env = HotpotQAEnvironment(knowledge_base=dict(full_kb or self.kb), feedback_mode='real') if backup else None

    def execute(self, action: str, action_input: str) -> str:
        a = action.strip().lower()
        if a == 'backup_search' and self.backup:
            obs = self._backup_env._search_real(action_input.strip())
            self.observation_meta.append({'obs_index': len(self.observation_meta) + 1, 'action': a,
                                          'regime_label': 'backup', 'corrupted': False, 'source': 'backup'})
            return obs
        if self.primary_disabled and a in ('search', 'lookup'):
            self.observation_meta.append({'obs_index': len(self.observation_meta) + 1, 'action': a,
                                          'regime_label': 'disabled', 'corrupted': True, 'source': 'primary'})
            return SWITCH_NOTICE
        obs = super().execute(action, action_input)
        if a != 'finish' and self.observation_meta:
            self.observation_meta[-1]['source'] = 'primary'
            if self.all_failed:
                self.observation_meta[-1]['corrupted'] = True
        return obs


class RuleSwitch:
    """Side-channel judgments on primary results; after k consecutive USELESS, disable the primary search."""

    def __init__(self, k: int = 5):
        self.k = k
        self.log: dict = {}

    def __call__(self, llm, undone):
        runs = [r for r in undone if r['trajectory'].steps and not r['env'].primary_disabled
                and (r['env'].observation_meta or [{}])[-1].get('source') == 'primary']
        if not runs:
            return
        outs = llm.batch_chat([r['messages'] + [{'role': 'user', 'content': JUDGE_PROMPT}] for r in runs], max_tokens=8, stop=[])
        for r, (text, _) in zip(runs, outs):
            j = parse_judgment(text)
            st = r.setdefault('switch_rule', {'k': 0})
            st['k'] = st['k'] + 1 if j == 'USELESS' else 0
            fired = st['k'] >= self.k
            if fired:
                r['env'].primary_disabled = True
                r['messages'][-1] = {'role': 'user', 'content': r['messages'][-1]['content'] + '\n\n' + SWITCH_NOTICE}
            tr = r['trajectory']
            self.log.setdefault((tr.question_id, tr.feedback_mode), []).append({
                'question_id': tr.question_id, 'feedback_mode': tr.feedback_mode, 't': len(tr.steps),
                'own_judgment_raw': text, 'own_judgment': j, 'k_useless_shown': st['k'],
                'decision': 'SWITCH' if fired else 'CONTINUE',
                'last_obs_corrupted': bool(r['env'].observation_meta[-1].get('corrupted'))})

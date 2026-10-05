"""Amendment 13 runner: answerless source, long recovery, and the two-source (backup search) setting.

Same questions (test300), decoding, step budget, arms and hooks as run_gate2.py; trajectories record the regime name
as feedback_mode (so logs are keyed by regime, not by environment mode).

Regimes:
  answerless            every page returned lacks the question's supporting-fact sentences (env_ext.answerless_kb)
  recover_after_4 / _5  the primary source fails on the first 4 / 5 observations, then recovers
  ts_<regime>           two-source setting: the <regime> failure schedule on the primary search, plus backup_search
                        (intact, never fails) described as slower and more expensive; <regime> in clean, persistent,
                        recover_after_1/2/3, late_onset_from_3
Arms: none, prompt_permit, prompt_budget, rule_k5_side, combo_budget_rule_k5, rule_switch_k5 (two-source only)
Usage: python run_gate2_ext.py --arm none --model qwen3-8b --gpu 0 --regimes answerless recover_after_4 [--num-questions N]
"""
from __future__ import annotations
import argparse, json, random
from collections import Counter
from datetime import datetime, timezone

from agent import AgentTrajectory, SYSTEM_PROMPT_V2, check_answer, trajectory_to_dict
from config import MODELS, SEED, stable_seed, MAX_AGENT_STEPS, BATCH_SIZE, SAMPLING_VERSION, SHUFFLE_MODE_DEFAULT
from data import load_run_data, download_hotpotqa
from inference import make_llm
from run_io import create_run_dir, write_json
import run_pilot
from gate2_intervention import DecisionStep
from prompt_arms import PromptArm, Combo, PROMPT_ARMS, COMBO_ARMS, PERMIT_SUFFIX
from env_ext import ExtEnvironment, RuleSwitch, answerless_kb, system_prompt_with_backup

BASE = {'clean': ('real', None), 'persistent': ('shuffled', None), 'recover_after_1': ('targeted', {1}),
        'recover_after_2': ('targeted', {1, 2}), 'recover_after_3': ('targeted', {1, 2, 3}),
        'late_onset_from_3': ('targeted', set(range(3, MAX_AGENT_STEPS + 1)))}
EXT_REGIMES = {'answerless': {'mode': 'real', 'corrupt': None, 'answerless': True, 'backup': False},
               'recover_after_4': {'mode': 'targeted', 'corrupt': {1, 2, 3, 4}, 'answerless': False, 'backup': False},
               'recover_after_5': {'mode': 'targeted', 'corrupt': {1, 2, 3, 4, 5}, 'answerless': False, 'backup': False}}
for _r, (_m, _c) in BASE.items():
    EXT_REGIMES['ts_' + _r] = {'mode': _m, 'corrupt': _c, 'answerless': False, 'backup': True}
# amendment 14: both exits named in the prompt (backup tool and answering from one's own knowledge)
for _r in ('clean', 'persistent', 'recover_after_2'):
    _m, _c = BASE[_r]
    EXT_REGIMES['tsboth_' + _r] = {'mode': _m, 'corrupt': _c, 'answerless': False, 'backup': True, 'permit': True}
ARMS = ['none', 'rule_switch_k5', 'rule_k5_side'] + list(PROMPT_ARMS) + list(COMBO_ARMS)


def make_hook(arm):
    if arm == 'none': return None
    if arm == 'rule_switch_k5': return RuleSwitch(5)
    if arm in COMBO_ARMS: return Combo(arm)
    if arm in PROMPT_ARMS: return PromptArm(arm)
    return DecisionStep(arm)


def run_condition(llm, questions, regime, pool, raw):
    spec = EXT_REGIMES[regime]
    rng = random.Random(stable_seed(SEED, 'gate2-ext', regime))
    out = []
    for b in range(0, len(questions), BATCH_SIZE):
        runs = []
        for q in questions[b:b + BATCH_SIZE]:
            kb = answerless_kb(q['knowledge_base'], raw[q['id']]) if spec['answerless'] else q['knowledge_base']
            env = ExtEnvironment(knowledge_base=kb, feedback_mode=spec['mode'], shuffle_pool=pool,
                                 rng=random.Random(rng.randint(0, 2**31)), corrupt_steps=spec['corrupt'] or set(),
                                 retrieval_backend='heuristic', oracle_titles=q.get('supporting_titles'), question_id=q['id'],
                                 shuffle_mode=SHUFFLE_MODE_DEFAULT, full_kb=q['knowledge_base'], backup=spec['backup'],
                                 all_failed=spec['answerless'])
            system = system_prompt_with_backup(SYSTEM_PROMPT_V2) if spec['backup'] else SYSTEM_PROMPT_V2
            if spec.get('permit'):
                system += PERMIT_SUFFIX
            msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': f"Question: {q['question']}"}]
            tr = AgentTrajectory(question_id=q['id'], question=q['question'], gold_answer=q['answer'], feedback_mode=regime,
                                 sampling_version=SAMPLING_VERSION, shuffle_mode=SHUFFLE_MODE_DEFAULT)
            runs.append({'env': env, 'messages': msgs, 'trajectory': tr, 'done': False})
        for _ in range(MAX_AGENT_STEPS):
            if all(r['done'] for r in runs):
                break
            runs = run_pilot.run_batch_step(llm, runs)
        for r in runs:
            tr = r['trajectory']
            if tr.final_answer is not None:
                tr.success = check_answer(tr.final_answer, tr.gold_answer)
            if tr.termination_reason is None:
                tr.termination_reason = 'budget_exhausted'
            tr.messages = r['messages']; tr.observation_meta = r['env'].observation_meta
            tr.shuffled_sources = r['env'].shuffled_sources
            out.append(tr)
    print(f'  {regime}: {sum(bool(t.success) for t in out)}/{len(out)}', flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', required=True, choices=ARMS)
    p.add_argument('--model', choices=list(MODELS), default='qwen3-8b'); p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--split', choices=['test300'], default='test300'); p.add_argument('--num-questions', type=int, default=None)
    p.add_argument('--regimes', nargs='+', required=True, choices=list(EXT_REGIMES)); p.add_argument('--tag', default='')
    args = p.parse_args()
    if args.arm == 'rule_switch_k5' and not all(EXT_REGIMES[r]['backup'] for r in args.regimes):
        raise SystemExit('rule_switch_k5 applies to two-source regimes only')
    questions, pool, meta = load_run_data('test', None, SHUFFLE_MODE_DEFAULT)
    if args.num_questions:
        questions = questions[:args.num_questions]
    meta = {**meta, 'question_ids': [q['id'] for q in questions], 'ext_regimes': {r: {k: (sorted(v) if isinstance(v, set) else v) for k, v in EXT_REGIMES[r].items()} for r in args.regimes}}
    raw = {x['_id']: x for x in download_hotpotqa()}
    out = create_run_dir(f'{args.model}_gate2ext_{args.arm}{("_" + args.tag) if args.tag else ""}', args, meta)
    print('RUN_DIRECTORY:', out, flush=True)
    llm = make_llm(MODELS[args.model], str(args.gpu))
    hook = make_hook(args.arm)
    run_pilot.PRE_ACTION_HOOK = hook
    summary = {}
    for regime in args.regimes:
        trs = run_condition(llm, questions, regime, pool, raw)
        write_json(out / f'{regime}.json', [dict(trajectory_to_dict(t), regime=regime) for t in trs])
        n_backup = [sum(1 for m in (t.observation_meta or []) if m.get('source') == 'backup') for t in trs]
        summary[regime] = {'n': len(trs), 'success': sum(bool(t.success) for t in trs),
                           'terminations': dict(Counter(t.termination_reason for t in trs)),
                           'used_backup': sum(1 for n in n_backup if n), 'backup_calls': sum(n_backup)}
        print(regime, json.dumps(summary[regime]), flush=True)
        if hook is not None and hasattr(hook, 'log'):
            (out / 'gate2_log.json').write_text(json.dumps([x for v in hook.log.values() for x in v], indent=1) + '\n')
    write_json(out / 'collection_summary.json', summary)
    mp = out / 'run_manifest.json'; m = json.loads(mp.read_text())
    m.update(status='completed', completed_utc=datetime.now(timezone.utc).isoformat(), generation_config=llm.generation_config,
             gate2_arm=args.arm, max_agent_steps=MAX_AGENT_STEPS)
    mp.write_text(json.dumps(m, indent=2) + '\n')
    print('COLLECTION_COMPLETE:', out, flush=True)


if __name__ == '__main__':
    main()

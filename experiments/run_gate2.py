"""Main collection script: runs one condition (--arm) for one model on a question split, in every failure regime.

Writes <run dir>/<regime>.json with the trajectories and <run dir>/gate2_log.json with per-step judgment and decision
records (side-channel conditions).

Usage: python run_gate2.py --arm <condition> --model qwen3-8b --split test300 --gpu 0 [--regimes ...]
       [--num-questions N] [--thinking] [--max-gen-tokens N]   (condition names: see README.md)
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from datetime import datetime, timezone
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))   # agent, environment and model modules
from agent import trajectory_to_dict
from config import MODELS, NUM_RUBRIC_QUESTIONS, SHUFFLE_MODE_DEFAULT, MAX_AGENT_STEPS
from data import load_run_data
from inference import make_llm
from run_io import create_run_dir, write_json
import run_pilot
from run_pilot import run_condition as run_standard
from run_targeted import run_condition as run_targeted
from gate2_intervention import DecisionStep
from prompt_arms import PROMPT_ARMS, PromptArm, COMBO_ARMS, Combo
from stated_rule import STATED_ARMS, StatedRule

REGIMES = [('clean', 'real'), ('persistent', 'shuffled'), ('plausible', 'plausible'), ('recover_after_2', 'targeted'),
           ('recover_after_1', 'targeted'), ('recover_after_3', 'targeted'), ('late_onset_from_3', 'targeted')]
CORRUPT = {'recover_after_1': {1}, 'recover_after_2': {1, 2}, 'recover_after_3': {1, 2, 3},
           'late_onset_from_3': set(range(3, MAX_AGENT_STEPS + 1))}  # {3..8} at the default budget


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', required=True, choices=sorted(__import__('gate2_intervention').ARMS) + ['none'] + list(PROMPT_ARMS) + list(COMBO_ARMS) + list(STATED_ARMS))
    p.add_argument('--model', choices=list(MODELS), default='qwen3-8b'); p.add_argument('--gpu', type=str, default='0', help='GPU id, or comma-separated ids for tensor parallelism')
    p.add_argument('--split', choices=['dev20', 'audit80', 'dev100', 'test300', 'fresh300'], default='audit80'); p.add_argument('--num-questions', type=int, default=None)
    p.add_argument('--regimes', nargs='*', default=None); p.add_argument('--thinking', action='store_true')
    p.add_argument('--max-gen-tokens', type=int, default=None); p.add_argument('--tag', default='')
    args = p.parse_args()
    if args.split == 'dev20':
        n = args.num_questions or NUM_RUBRIC_QUESTIONS
        questions, pool, meta = load_run_data('dev', n, SHUFFLE_MODE_DEFAULT)
    elif args.split == 'test300':
        questions, pool, meta = load_run_data('test', None, SHUFFLE_MODE_DEFAULT)
        if args.num_questions:
            questions = questions[:args.num_questions]
        meta = {**meta, 'question_ids': [q['id'] for q in questions]}
    elif args.split == 'fresh300':
        questions, pool, meta = load_run_data('fresh', None, SHUFFLE_MODE_DEFAULT)
        if args.num_questions:
            questions = questions[:args.num_questions]
        meta = {**meta, 'question_ids': [q['id'] for q in questions]}
    elif args.split == 'dev100':
        questions, pool, meta = load_run_data('dev', None, SHUFFLE_MODE_DEFAULT)
    else:
        full, pool, meta = load_run_data('dev', None, SHUFFLE_MODE_DEFAULT)
        questions = full[NUM_RUBRIC_QUESTIONS:]
        if args.num_questions:
            questions = questions[:args.num_questions]
        meta = {**meta, 'question_ids': [q['id'] for q in questions], 'split_role': 'reserved_audit80'}
    regimes = [(r, m) for r, m in REGIMES if not args.regimes or r in args.regimes]
    out = create_run_dir(f'{args.model}_gate2_{args.arm}{("_" + args.tag) if args.tag else ""}', args, meta)
    print('RUN_DIRECTORY:', out, flush=True)
    write_json(out / 'questions_private.json', questions)
    llm = make_llm(MODELS[args.model], str(args.gpu), args.thinking, args.max_gen_tokens)
    if args.arm in STATED_ARMS:
        hook = StatedRule(args.arm)   # amendment 19: rule on the judgment stated in the agent's own reasoning
        run_pilot.POST_GENERATION_HOOK = hook
    elif args.arm in COMBO_ARMS:
        hook = Combo(args.arm)  # budget prompt + side-channel rule (addendum 6)
    elif args.arm in PROMPT_ARMS:
        hook = PromptArm(args.arm)  # prompt-only control: edits text the agent sees, no judgment/decision calls
    else:
        hook = DecisionStep(args.arm) if args.arm != 'none' else DecisionStep('own')  # 'none' = baseline, hook not installed
    run_pilot.PRE_ACTION_HOOK = (hook.pre() if args.arm in STATED_ARMS else hook) if args.arm != 'none' else None
    summary = {}
    for regime, mode in regimes:
        if mode == 'targeted':
            trajectories = run_targeted(llm, questions, regime, CORRUPT[regime], pool, 'heuristic', shuffle_mode=SHUFFLE_MODE_DEFAULT)
        else:
            trajectories = run_standard(llm, questions, mode, pool, 'heuristic', shuffle_mode=SHUFFLE_MODE_DEFAULT)
        records = [dict(trajectory_to_dict(t), regime=regime) for t in trajectories]
        write_json(out / f'{regime}.json', records)
        logs = [x for t in trajectories for x in hook.log.get((t.question_id, t.feedback_mode), [])]
        dec = Counter(x['decision'] for x in logs)
        summary[regime] = {'n': len(trajectories), 'success': sum(bool(t.success) for t in trajectories),
                           'terminations': dict(Counter(t.termination_reason for t in trajectories)),
                           'decision_points': len(logs), 'decisions': dict(dec),
                           'own_judgment': dict(Counter(x['own_judgment'] for x in logs)),
                           'finish_after_answer_now': sum(1 for t in trajectories for x in hook.log.get((t.question_id, t.feedback_mode), [])
                                                          if x['decision'] == 'ANSWER_NOW' and x['t'] < len(t.steps) and t.steps[x['t']].action == 'finish')}
        print(regime, json.dumps(summary[regime]), flush=True)
        # persist judgments after every regime so an interrupted run keeps the finished regimes' logs
        (out / 'gate2_log.json').write_text(json.dumps([x for v in hook.log.values() for x in v], indent=1) + '\n')
    write_json(out / 'collection_summary.json', summary)
    mp = out / 'run_manifest.json'; m = json.loads(mp.read_text())
    m.update(status='completed', completed_utc=datetime.now(timezone.utc).isoformat(), generation_config=llm.generation_config, gate2_arm=args.arm, max_agent_steps=MAX_AGENT_STEPS, api_usage_totals=getattr(llm, 'usage_totals', None), stated_regenerations=getattr(hook, 'regenerated', None))
    mp.write_text(json.dumps(m, indent=2) + '\n')
    print('COLLECTION_COMPLETE:', out, flush=True)


if __name__ == '__main__':
    main()

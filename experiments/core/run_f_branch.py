"""Answer-now branches: the F branch ("answer now, no further tool calls") from every decision point.

For each saved v2 trajectory and each decision point t (right after the t-th tool observation),
the recorded history through that observation is replayed verbatim and the F protocol message is
appended IN THE SAME USER SLOT as the observation. The model then produces one
turn; a finish[...] action is scored with the same check_answer as the main runs.

Because the main runs are deterministic given the prompt, the C branch (continue) at decision
point t is the recorded trajectory itself; only F needs new generations.

Usage:
  python run_f_branch.py --run-dir <v2 run dir> --gpu 3 [--protocol main|alt] [--regimes clean persistent ...]
                         [--limit-trajectories N] [--dry-run]
Outputs a new run dir results_v2/<stamp>_<model>_fbranch_<protocol>/ with one JSON per regime:
  {question_id, regime, decision_index (1-based obs count), prompt_messages_sha256, raw_response,
   parsed_action, answer, success, gold_answer}
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from agent import parse_agent_response, check_answer
from config import MODELS, F_PROTOCOL_MAIN, F_PROTOCOL_ALT, F_PROTOCOL_STRICT_SYSTEM_SUFFIX, RESULTS_DIR_V2
from run_io import create_run_dir, complete_run, write_json


def decision_points(traj):
    """Yield (t, messages_through_obs_t) for t = 1..n_obs using the recorded message list."""
    msgs = traj['messages']
    # layout: system, user(question), then (assistant, user(observation)) pairs
    n_pairs = (len(msgs) - 2) // 2
    for t in range(1, n_pairs + 1):
        cut = 2 + 2 * t
        last = msgs[cut - 1]
        if last['role'] != 'user' or not last['content'].startswith('Observation:'):
            continue
        if last['content'].startswith('Observation: Answer submitted'):
            continue  # post-finish message; no decision follows
        yield t, msgs[:cut]


def build_f_prompt(history, protocol_text, strict=False):
    prompt = [dict(m) for m in history]
    prompt[-1] = {'role': 'user', 'content': prompt[-1]['content'] + '\n\n' + protocol_text}
    if strict:
        prompt[0] = {'role': 'system', 'content': prompt[0]['content'] + F_PROTOCOL_STRICT_SYSTEM_SUFFIX}
    return prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--model', default=None, help='defaults to the model recorded in the run manifest')
    p.add_argument('--protocol', choices=['main', 'alt', 'strict'], default='main')
    p.add_argument('--regimes', nargs='*', default=None)
    p.add_argument('--limit-trajectories', type=int, default=None)
    p.add_argument('--offset-trajectories', type=int, default=0, help='skip the first N trajectories per regime (staged runs)')
    p.add_argument('--thinking', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='count prompts; no model load')
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / 'run_manifest.json').read_text())
    if manifest.get('status') != 'completed':
        raise SystemExit('source run is not completed')
    model_key = args.model or manifest['arguments']['model']
    protocol_text = F_PROTOCOL_ALT if args.protocol == 'alt' else F_PROTOCOL_MAIN
    regimes = args.regimes or [f.stem for f in sorted(run_dir.glob('*.json'))
                               if f.stem not in ('run_manifest', 'collection_summary', 'quality_checks', 'questions_private')]

    jobs = []  # (regime, traj, t, prompt)
    for regime in regimes:
        trajs = json.loads((run_dir / f'{regime}.json').read_text())
        trajs = trajs[args.offset_trajectories:]
        if args.limit_trajectories:
            trajs = trajs[:args.limit_trajectories]
        for traj in trajs:
            for t, hist in decision_points(traj):
                jobs.append((regime, traj, t, build_f_prompt(hist, protocol_text, strict=(args.protocol == 'strict'))))
    counts = Counter(j[0] for j in jobs)
    print(json.dumps({'source_run': manifest['run_id'], 'model': model_key, 'protocol': args.protocol,
                      'decision_points': dict(counts), 'total': len(jobs)}, indent=2), flush=True)
    if args.dry_run:
        return

    from inference import make_llm
    out = create_run_dir(f'{model_key}_fbranch_{args.protocol}', args,
                         {'source_run': manifest['run_id'], 'source_manifest_sha256':
                          hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                          'protocol_text': protocol_text})
    print('RUN_DIRECTORY:', out, flush=True)
    llm = make_llm(MODELS[model_key], str(args.gpu), args.thinking)
    results = {r: [] for r in regimes}
    B = 64
    for i in range(0, len(jobs), B):
        chunk = jobs[i:i + B]
        outs = llm.batch_chat([j[3] for j in chunk])
        for (regime, traj, t, prompt), o in zip(chunk, outs):
            text = o['text'] if isinstance(o, dict) else o[0]
            thought, action, action_input = parse_agent_response(text)
            answered = action == 'finish' and bool(action_input.strip())
            results[regime].append({
                'question_id': traj['question_id'], 'regime': regime, 'decision_index': t,
                'remaining_action_slots_at_t': max(0, 8 - t),
                'prompt_messages_sha256': hashlib.sha256(json.dumps(prompt, sort_keys=True).encode()).hexdigest(),
                'raw_response': text, 'parsed_action': action, 'answer': action_input if answered else None,
                'success': bool(answered and check_answer(action_input, traj['gold_answer'])),
                'gold_answer': traj['gold_answer'],
                'c_branch_success': traj['success'],  # recorded trajectory = continue branch outcome
                'c_branch_termination': traj['termination_reason'],
            })
        print(f'  {i + len(chunk)}/{len(jobs)}', flush=True)
    for regime, rows in results.items():
        write_json(out / f'{regime}.json', rows)
    summary = {r: {'n': len(rows), 'answered': sum(x['answer'] is not None for x in rows),
                   'f_success': sum(x['success'] for x in rows),
                   'by_decision_index': {str(t): [sum(1 for x in rows if x['decision_index'] == t),
                                                  sum(x['success'] for x in rows if x['decision_index'] == t)]
                                         for t in sorted({x['decision_index'] for x in rows})}}
               for r, rows in results.items()}
    write_json(out / 'summary.json', summary)
    complete_run(out, llm.generation_config)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()

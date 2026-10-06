"""Addendum 12a: replay the side-channel judgment on recorded trajectories.

At every decision point of a saved run (after the t-th observation), the recorded history is replayed verbatim and the
rule arm's side-channel question (gate2_intervention.JUDGE_PROMPT, at most 8 tokens) is appended. The reply is
recorded; nothing is written back, so the trajectories themselves are untouched. This gives, for arms that never
asked for judgments (none, prompt arms), the agent's own USEFUL/USELESS judgment on exactly the trajectory it acted in.

Usage: python run_judge_replay.py --run-dir results_v2/<run> --gpu 0 [--regimes ...] [--limit-trajectories N]
Output: results_v2/<stamp>_<model>_judgereplay_<arm>/<regime>.json with {question_id, regime, t, raw, judgment}.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))   # agent, environment and model modules
from config import MODELS
from gate2_intervention import JUDGE_PROMPT, parse_judgment
from run_f_branch import decision_points
from run_io import create_run_dir, write_json

FAIL_REGIMES = ['persistent', 'recover_after_1', 'recover_after_2', 'recover_after_3', 'late_onset_from_3']
CHUNK = 512


def build_prompts(trajs):
    """(meta, messages) for every decision point; messages = recorded history + the side-channel question."""
    out = []
    for tr in trajs:
        for t, hist in decision_points(tr):
            out.append(({'question_id': tr['question_id'], 't': t},
                        [dict(m) for m in hist] + [{'role': 'user', 'content': JUDGE_PROMPT}]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--regimes', nargs='*', default=None)
    p.add_argument('--limit-trajectories', type=int, default=None)
    args, _ = p.parse_known_args()   # extra arguments such as --arm/--model/--tag are ignored
    src = Path(args.run_dir)
    man = json.loads((src / 'run_manifest.json').read_text())
    model_key = man['arguments']['model']
    arm = man.get('gate2_arm') or man['arguments'].get('arm')
    regimes = [r for r in (args.regimes or FAIL_REGIMES) if (src / f'{r}.json').exists()]
    out = create_run_dir(f'{model_key}_judgereplay_{arm}', args,
                         {'source_run': str(src), 'source_manifest_sha256': hashlib.sha256((src / 'run_manifest.json').read_bytes()).hexdigest(),
                          'judge_prompt': JUDGE_PROMPT, 'regimes': regimes})
    print('RUN_DIRECTORY:', out, flush=True)
    from inference import make_llm
    llm = make_llm(MODELS[model_key], args.gpu)
    summary = {}
    for reg in regimes:
        trajs = json.loads((src / f'{reg}.json').read_text())[:args.limit_trajectories]
        items = build_prompts(trajs)
        rows = []
        for i in range(0, len(items), CHUNK):
            chunk = items[i:i + CHUNK]
            outs = llm.batch_chat([m for _, m in chunk], max_tokens=8, stop=[])
            for (meta, _), (text, _) in zip(chunk, outs):
                rows.append({**meta, 'regime': reg, 'raw': text, 'judgment': parse_judgment(text)})
        write_json(out / f'{reg}.json', rows)
        summary[reg] = {'n': len(rows), 'useless': sum(r['judgment'] == 'USELESS' for r in rows),
                        'unparsed': sum(r['judgment'] == 'UNPARSED' for r in rows)}
        print(reg, json.dumps(summary[reg]), flush=True)
    write_json(out / 'summary.json', summary)
    mp = out / 'run_manifest.json'; m = json.loads(mp.read_text())
    m.update(status='completed', generation_config=llm.generation_config)
    mp.write_text(json.dumps(m, indent=2) + '\n')
    print('COLLECTION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

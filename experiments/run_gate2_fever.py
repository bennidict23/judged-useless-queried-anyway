"""Gate 2 transfer to a second task: FEVER claim verification (addendum 4; the pre-registration).

Same environment, regimes, arms, step budget and decoding as run_gate2.py; only the task changes:
  - data: the cached FEVER set (fever_data.json, 292 claims, 146 SUPPORTS / 146 REFUTES), per-question v2 shuffle pool
  - system prompt: FEVER verdict prompt (+ the v2 one-action-per-turn sentence used on HotpotQA)
  - task wording in the hook prompts: "claim"/"verdict" instead of "question"/"answer"
  - scoring: strict verdict match (exactly one of SUPPORTS / REFUTES present, equal to gold)
No parameter is selected on FEVER: k=5, lexical tau and rate-matched p are carried over from HotpotQA dev.

Usage: python run_gate2_fever.py --arm none|rule_k5_side|... --model qwen3-8b --gpu 0 [--regimes ...] [--num-questions N] [--tag T]
"""
from __future__ import annotations
import argparse, hashlib, json, random, re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from agent import AgentTrajectory, SYSTEM_PROMPT_V2, SYSTEM_PROMPT, trajectory_to_dict
from config import MODELS, SEED, stable_seed, MAX_AGENT_STEPS, BATCH_SIZE, SAMPLING_VERSION, SHUFFLE_MODE_DEFAULT
from data import build_shuffle_pool_v2
from inference import make_llm
from run_io import create_run_dir, write_json
from tools import HotpotQAEnvironment
import run_pilot
import gate2_intervention as g2
from gate2_intervention import DecisionStep
from prompt_arms import PROMPT_ARMS, PromptArm, COMBO_ARMS, Combo
from run_gate2 import REGIMES, CORRUPT

FEVER_CACHE = Path(__file__).parent / 'fever_data.json'

V2_SENTENCE = SYSTEM_PROMPT_V2[len(SYSTEM_PROMPT):]  # "Produce exactly one Thought and one Action per turn..."
FEVER_SYSTEM_PROMPT = """\
You are a fact verification agent. You verify claims by searching a Wikipedia-like knowledge base.

Available actions:
- search[query]: Search for a Wikipedia article. Returns the article text.
- lookup[keyword]: Find the next sentence containing the keyword in the current article.
- finish[answer]: Submit your verdict. The answer must be exactly one of: SUPPORTS, REFUTES

Format your response as:
Thought: <your step-by-step reasoning about whether the claim is supported or refuted>
Action: <action_name>
Action Input: <input>""" + V2_SENTENCE

# task wording for the hook prompts (module globals are read at call time)
FEVER_WORDING = {
    'JUDGE_PROMPT': ("Before acting: was the LAST observation useful for verifying the original claim? "
                     "Reply with exactly one word: USEFUL or USELESS."),
    'F_PROTOCOL_MAIN': "Do not make any further tool calls. Give your final verdict on the original claim using finish[answer].",
    'ACT_PROMPT': ("Now continue in the required format (Thought / Action / Action Input), acting consistently with your "
                   "decision: CONTINUE means a search or lookup action; ANSWER_NOW means the finish action with your verdict "
                   "(SUPPORTS or REFUTES) on the original claim as the Action Input."),
}


def apply_fever_wording():
    for k, v in FEVER_WORDING.items():
        assert hasattr(g2, k), k
        setattr(g2, k, v)


def fever_verdict(pred: str | None) -> str | None:
    """Return SUPPORTS / REFUTES if exactly one verdict is expressed, else None."""
    if not pred:
        return None
    t = pred.upper()
    has_s, has_r = bool(re.search(r'SUPPORT', t)), bool(re.search(r'REFUT', t))
    if has_s == has_r:
        return None
    return 'SUPPORTS' if has_s else 'REFUTES'


def check_fever_strict(pred: str | None, gold: str) -> bool:
    return fever_verdict(pred) == gold.strip().upper()


def load_fever():
    qs = json.loads(FEVER_CACHE.read_text())
    for q in qs:
        q['distractor_titles'] = [t for t in q['knowledge_base'] if t not in set(q['supporting_titles'])]
    pool = build_shuffle_pool_v2(qs)
    meta = {'question_set': 'fever292', 'question_ids': [q['id'] for q in qs],
            'dataset_sha256': hashlib.sha256(FEVER_CACHE.read_bytes()).hexdigest(),
            'pool_sha256': hashlib.sha256(json.dumps(pool, sort_keys=True).encode()).hexdigest(),
            'split_role': 'fever_transfer_test (no parameter selected on FEVER)'}
    return qs, pool, meta


def run_condition(llm, questions, regime, mode, corrupt_steps, pool):
    rng = random.Random(stable_seed(SEED, 'fever-gate2', regime))
    out = []
    for b in range(0, len(questions), BATCH_SIZE):
        runs = []
        for q in questions[b:b + BATCH_SIZE]:
            env = HotpotQAEnvironment(
                knowledge_base=q['knowledge_base'], feedback_mode=mode, shuffle_pool=pool,
                rng=random.Random(rng.randint(0, 2**31)), corrupt_steps=corrupt_steps or set(),
                retrieval_backend='heuristic', oracle_titles=q.get('supporting_titles'),
                distractor_titles=q.get('distractor_titles'), question_id=q['id'], shuffle_mode=SHUFFLE_MODE_DEFAULT)
            msgs = [{'role': 'system', 'content': FEVER_SYSTEM_PROMPT},
                    {'role': 'user', 'content': f"Verify this claim: {q['question']}"}]
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
                tr.success = check_fever_strict(tr.final_answer, tr.gold_answer)
            if tr.termination_reason is None:
                tr.termination_reason = 'budget_exhausted'
            tr.messages = r['messages']; tr.observation_meta = r['env'].observation_meta
            tr.shuffled_sources = r['env'].shuffled_sources
            out.append(tr)
    print(f'  {regime}: {sum(bool(t.success) for t in out)}/{len(out)}', flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', required=True, choices=sorted(g2.ARMS) + ['none'] + list(PROMPT_ARMS) + list(COMBO_ARMS))
    p.add_argument('--model', choices=list(MODELS), default='qwen3-8b'); p.add_argument('--gpu', type=str, default='0', help='GPU id, or comma-separated ids for tensor parallelism')
    p.add_argument('--regimes', nargs='*', default=None); p.add_argument('--num-questions', type=int, default=None)
    p.add_argument('--thinking', action='store_true'); p.add_argument('--max-gen-tokens', type=int, default=None)
    p.add_argument('--tag', default='')
    args = p.parse_args()
    apply_fever_wording()
    questions, pool, meta = load_fever()
    if args.num_questions:
        questions = questions[:args.num_questions]
        meta = {**meta, 'question_ids': [q['id'] for q in questions]}
    regimes = [(r, m) for r, m in REGIMES if not args.regimes or r in args.regimes]
    out = create_run_dir(f'{args.model}_fever_gate2_{args.arm}{("_" + args.tag) if args.tag else ""}', args, meta)
    print('RUN_DIRECTORY:', out, flush=True)
    write_json(out / 'questions_private.json', [{k: q[k] for k in ('id', 'question', 'answer')} for q in questions])
    llm = make_llm(MODELS[args.model], str(args.gpu), args.thinking, args.max_gen_tokens)
    if args.arm in COMBO_ARMS:
        hook = Combo(args.arm)
    elif args.arm in PROMPT_ARMS:
        hook = PromptArm(args.arm)
    else:
        hook = DecisionStep(args.arm) if args.arm != 'none' else DecisionStep('own')
    run_pilot.PRE_ACTION_HOOK = hook if args.arm != 'none' else None
    summary = {}
    for regime, mode in regimes:
        trs = run_condition(llm, questions, regime, mode, CORRUPT.get(regime) if mode == 'targeted' else None, pool)
        write_json(out / f'{regime}.json', [dict(trajectory_to_dict(t), regime=regime) for t in trs])
        logs = [x for t in trs for x in hook.log.get((t.question_id, t.feedback_mode), [])]
        summary[regime] = {'n': len(trs), 'success': sum(bool(t.success) for t in trs),
                           'terminations': dict(Counter(t.termination_reason for t in trs)),
                           'verdict_unparsed': sum(1 for t in trs if t.final_answer is not None and fever_verdict(t.final_answer) is None),
                           'decisions': dict(Counter(x['decision'] for x in logs)),
                           'own_judgment': dict(Counter(x['own_judgment'] for x in logs))}
        print(regime, json.dumps(summary[regime]), flush=True)
        (out / 'gate2_log.json').write_text(json.dumps([x for v in hook.log.values() for x in v], indent=1) + '\n')
    write_json(out / 'collection_summary.json', summary)
    mp = out / 'run_manifest.json'; m = json.loads(mp.read_text())
    m.update(status='completed', completed_utc=datetime.now(timezone.utc).isoformat(), generation_config=llm.generation_config,
             gate2_arm=args.arm, task='fever', max_agent_steps=MAX_AGENT_STEPS, fever_wording=FEVER_WORDING, api_usage_totals=getattr(llm, 'usage_totals', None))
    mp.write_text(json.dumps(m, indent=2) + '\n')
    print('COLLECTION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

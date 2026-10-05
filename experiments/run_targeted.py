"""Observation replacement at chosen steps: used by run_gate2.py for the recover_after_c and late_onset_from_3
regimes (only the listed observations are replaced; all others are real).
"""

from __future__ import annotations

import argparse
import json
import random
import time

from config import (
    MODELS, MAX_AGENT_STEPS, MAX_GEN_TOKENS, BATCH_SIZE, RESULTS_DIR, RESULTS_DIR_V2, SEED,
    SAMPLING_VERSION, SHUFFLE_MODE_DEFAULT, stable_seed,
)
from data import load_run_data
from run_io import create_run_dir, complete_run
from run_pilot import run_batch_step
from tools import HotpotQAEnvironment
from agent import (
    SYSTEM_PROMPT, SYSTEM_PROMPT_V2, AgentTrajectory, AgentStep,
    parse_agent_response, check_answer, trajectory_to_dict,
)
from inference import BatchLocalLLM

TARGETED_CONDITIONS = {
    "real":        {"corrupt_steps": set()},
    "corrupt_1":   {"corrupt_steps": {1}},
    "corrupt_2":   {"corrupt_steps": {2}},
    "corrupt_3":   {"corrupt_steps": {3}},
    "corrupt_1_2": {"corrupt_steps": {1, 2}},
    "corrupt_all": {"corrupt_steps": {1, 2, 3, 4, 5, 6, 7, 8}},
}

# v2 regimes, counted in returned tool observations (1-indexed):
#   persistent          = corrupt_all
#   recover_after_k     = first k observations misleading, then the clean backend
#   late_onset_from_k   = clean until observation k-1, misleading from k on
REGIME_CONDITIONS = {
    "persistent":        {"corrupt_steps": {1, 2, 3, 4, 5, 6, 7, 8}},
    "recover_after_1":   {"corrupt_steps": {1}},
    "recover_after_2":   {"corrupt_steps": {1, 2}},
    "recover_after_3":   {"corrupt_steps": {1, 2, 3}},
    "late_onset_from_3": {"corrupt_steps": {3, 4, 5, 6, 7, 8}},
    "clean":             {"corrupt_steps": set()},
}


def run_condition(llm, questions, cond_name, corrupt_steps, shuffle_pool, retrieval_backend,
                  shuffle_mode: str = "legacy"):
    """shuffle_mode defaults to legacy for backward compatibility;
    pass shuffle_mode="per_question" (and a build_shuffle_pool_v2 pool) for v2 runs."""
    print(f"\n  {cond_name}: corrupt steps = {sorted(corrupt_steps) if corrupt_steps else 'none'}")
    print(f"    retrieval backend = {retrieval_backend}, shuffle_mode = {shuffle_mode}")

    rng = random.Random(stable_seed(SEED, "targeted", cond_name))
    all_trajectories = []

    for batch_start in range(0, len(questions), BATCH_SIZE):
        batch_qs = questions[batch_start:batch_start + BATCH_SIZE]

        active_runs = []
        for q in batch_qs:
            env = HotpotQAEnvironment(
                knowledge_base=q["knowledge_base"],
                feedback_mode="targeted",
                shuffle_pool=shuffle_pool,
                rng=random.Random(rng.randint(0, 2**31)),
                corrupt_steps=corrupt_steps,
                retrieval_backend=retrieval_backend,
                oracle_titles=q.get("supporting_titles"),
                question_id=q["id"],
                shuffle_mode=shuffle_mode,
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_V2 if shuffle_mode == "per_question" else SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {q['question']}"},
            ]
            trajectory = AgentTrajectory(
                question_id=q["id"],
                question=q["question"],
                gold_answer=q["answer"],
                feedback_mode=cond_name,
                sampling_version=SAMPLING_VERSION if shuffle_mode == "per_question" else "v1",
                shuffle_mode=shuffle_mode,
            )
            active_runs.append({
                "env": env, "messages": messages,
                "trajectory": trajectory, "done": False,
            })

        for _ in range(MAX_AGENT_STEPS):
            if not any(not r["done"] for r in active_runs):
                break
            active_runs = run_batch_step(llm, active_runs)

        for run in active_runs:
            traj = run["trajectory"]
            if traj.final_answer is not None:
                traj.success = check_answer(traj.final_answer, traj.gold_answer)
            if traj.termination_reason is None:
                traj.termination_reason = "budget_exhausted"
            traj.messages = run["messages"]
            traj.observation_meta = run["env"].observation_meta
            traj.shuffled_sources = run["env"].shuffled_sources
            all_trajectories.append(traj)

    successes = sum(1 for t in all_trajectories if t.success)
    total = len(all_trajectories)
    print(f"    Results: {successes}/{total} = {successes/total:.1%}")
    return all_trajectories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-gen-tokens", type=int, default=None)
    parser.add_argument(
        "--retrieval-backend",
        default="heuristic",
        choices=["heuristic", "oracle"],
    )
    parser.add_argument("--shuffle-mode", default=SHUFFLE_MODE_DEFAULT, choices=["per_question", "legacy"])
    parser.add_argument("--question-set", default="main", choices=["main", "dev"])
    parser.add_argument("--regimes", action="store_true",
                        help="run the v2 REGIME_CONDITIONS (persistent / recover_after_k / late_onset / clean) "
                             "instead of the paper's TARGETED_CONDITIONS")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    questions, shuffle_pool, data_meta = load_run_data(
        args.question_set, args.num_questions, args.shuffle_mode)
    out_dir = create_run_dir(f"{args.model}_targeted{('_'+args.tag) if args.tag else ''}", args, data_meta)
    llm = BatchLocalLLM(MODELS[args.model], gpu_ids=str(args.gpu), thinking=args.thinking,
                        max_tokens=args.max_gen_tokens)
    conditions = REGIME_CONDITIONS if args.regimes else TARGETED_CONDITIONS
    suffix = ("_dev" if args.question_set == "dev" else "") + (f"_{args.tag}" if args.tag else "")

    print(f"=== Step-Targeted Corruption — {args.model} ===")
    print(f"Questions: {len(questions)}")
    print(f"Retrieval backend: {args.retrieval_backend}")

    all_results = {}
    start = time.time()

    for cond_name, cond_config in conditions.items():
        t0 = time.time()
        trajectories = run_condition(
            llm,
            questions,
            cond_name,
            cond_config["corrupt_steps"],
            shuffle_pool,
            args.retrieval_backend,
            shuffle_mode=args.shuffle_mode,
        )
        elapsed = time.time() - t0

        all_results[cond_name] = {
            "corrupt_steps": sorted(cond_config["corrupt_steps"]),
            "success_rate": sum(1 for t in trajectories if t.success) / len(trajectories),
            "n": len(trajectories),
            "shuffle_mode": args.shuffle_mode,
            "question_set": args.question_set,
            "data": [trajectory_to_dict(t, full=True) for t in trajectories],
        }
        print(f"    ({elapsed:.0f}s)")

    backend_suffix = "" if args.retrieval_backend == "heuristic" else f"_{args.retrieval_backend}"
    family = "regimes" if args.regimes else "targeted"
    out = out_dir / f"{args.model}_{family}{backend_suffix}{suffix}.json"
    with open(out, "x") as f:
        json.dump(all_results, f, indent=2)

    complete_run(out_dir, llm.generation_config)
    total = time.time() - start
    print(f"\nTotal: {total:.0f}s ({total/60:.1f}min)")
    print(f"Saved: {out}")

    # Summary
    print(f"\n{'Condition':<15} {'Corrupt Steps':>15} {'Success':>10}")
    print("-" * 45)
    for name in conditions:
        r = all_results[name]
        steps_str = str(r["corrupt_steps"]) if r["corrupt_steps"] else "none"
        print(f"{name:<15} {steps_str:>15} {r['success_rate']:>9.1%}")


if __name__ == "__main__":
    main()

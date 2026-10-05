"""Agent loop driver shared by the collection scripts (HotpotQA, observation-replacement environment).

Three conditions, same prompt, only observation content differs:
  - real:        correct Wikipedia paragraphs
  - shuffled:    real paragraphs from WRONG questions
  - no_feedback: "no results found, reason from knowledge"

Usage:
    python run_pilot.py --model llama3.1-8b --gpu 0
    python run_pilot.py --model qwen2.5-7b --gpu 2
"""

from __future__ import annotations

import argparse
import json
import random
import time

from config import (
    MODELS, FEEDBACK_CONDITIONS, MAX_AGENT_STEPS,
    MAX_GEN_TOKENS, BATCH_SIZE, RESULTS_DIR, RESULTS_DIR_V2, SEED,
    DEFAULT_PILOT_CONDITIONS, NUM_QUESTIONS, SAMPLING_VERSION,
    SHUFFLE_MODE_DEFAULT, stable_seed,
)
from data import (
    load_run_data,
)
from tools import HotpotQAEnvironment
from agent import (
    SYSTEM_PROMPT, SYSTEM_PROMPT_V2, AgentTrajectory, AgentStep,
    parse_agent_response, check_answer, trajectory_to_dict,
)
from inference import BatchLocalLLM
from run_io import create_run_dir, complete_run


# Optional pre-action hook (Gate 2 interventions): called with (llm, undone_runs) before the action
# generation of every step. None = the unmodified agent loop.
PRE_ACTION_HOOK = None
# Optional post-generation hook (amendment 19): called with (llm, undone_runs, results, metadata) after the action
# generation and before any action is executed; returns (results, metadata), possibly with some turns regenerated.
# None = the unmodified agent loop (all earlier arms).
POST_GENERATION_HOOK = None


def run_batch_step(
    llm: BatchLocalLLM,
    active_runs: list[dict],
) -> list[dict]:
    """Run one agent step for all active runs in parallel."""
    undone = [r for r in active_runs if not r["done"]]
    if not undone:
        return active_runs
    if PRE_ACTION_HOOK is not None:
        PRE_ACTION_HOOK(llm, undone)

    batch_messages = [r["messages"] for r in undone]
    results = llm.batch_chat(batch_messages, max_tokens=getattr(llm, "max_tokens", MAX_GEN_TOKENS))

    metadata = getattr(llm, "last_batch_metadata", [{} for _ in results])
    if POST_GENERATION_HOOK is not None:
        results, metadata = POST_GENERATION_HOOK(llm, undone, results, metadata)
    for run, (response, num_tokens), meta in zip(undone, results, metadata):
        env = run["env"]
        trajectory = run["trajectory"]

        trajectory.generation_config = getattr(llm, "generation_config", {})
        trajectory.generations.append(dict(meta, raw_response=response))
        if meta.get("truncated"):
            trajectory.termination_reason = "generation_truncated"
            run["messages"].append({"role": "assistant", "content": response})
            run["done"] = True
            continue
        thought, action, action_input = parse_agent_response(
            response, strict=trajectory.shuffle_mode == "per_question")

        if not action:
            trajectory.termination_reason = "format_failure"
            run["messages"].append({"role": "assistant", "content": response})
            run["done"] = True
            continue

        # Track action types
        if action == "search":
            trajectory.num_searches += 1
        elif action == "lookup":
            trajectory.num_lookups += 1

        observation = env.execute(action, action_input)

        step = AgentStep(
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
            thought_tokens=len(thought.split()) if thought else 0,
            raw_response=response,
        )
        trajectory.steps.append(step)
        run["messages"].append({"role": "assistant", "content": response})
        run["messages"].append({
            "role": "user",
            "content": f"Observation: {observation}",
        })

        if env.finished:
            trajectory.final_answer = env.final_answer
            trajectory.termination_reason = "answered"
            run["done"] = True
            continue

        if len(trajectory.steps) >= MAX_AGENT_STEPS:
            trajectory.termination_reason = "budget_exhausted"
            run["done"] = True
            continue

    return active_runs


def run_condition(
    llm: BatchLocalLLM,
    questions: list[dict],
    condition: str,
    shuffle_pool: list,
    retrieval_backend: str,
    shuffle_mode: str = SHUFFLE_MODE_DEFAULT,
) -> list[AgentTrajectory]:
    """Run all questions under one feedback condition."""
    print(f"\n{'='*60}")
    print(f"  Feedback condition: {condition}")
    print(f"  Retrieval backend: {retrieval_backend}")
    print(f"  {FEEDBACK_CONDITIONS[condition]}")
    print(f"  Questions: {len(questions)}")
    print(f"{'='*60}")

    rng = random.Random(stable_seed(SEED, condition))  # v2: PYTHONHASHSEED-independent
    all_trajectories = []

    for batch_start in range(0, len(questions), BATCH_SIZE):
        batch_qs = questions[batch_start:batch_start + BATCH_SIZE]
        print(f"  Batch {batch_start // BATCH_SIZE + 1}: "
              f"questions {batch_start+1}-{batch_start+len(batch_qs)}")

        active_runs = []
        for q in batch_qs:
            # Create environment with the specified feedback mode
            # For shuffled: use paragraphs from OTHER questions
            env = HotpotQAEnvironment(
                knowledge_base=q["knowledge_base"],
                feedback_mode=condition,
                shuffle_pool=shuffle_pool,
                rng=random.Random(rng.randint(0, 2**31)),
                retrieval_backend=retrieval_backend,
                oracle_titles=q.get("supporting_titles"),
                distractor_titles=q.get("distractor_titles"),
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
                feedback_mode=condition,
                sampling_version=SAMPLING_VERSION if shuffle_mode == "per_question" else "v1",
                shuffle_mode=shuffle_mode,
            )

            active_runs.append({
                "env": env,
                "messages": messages,
                "trajectory": trajectory,
                "done": False,
            })

        for step_idx in range(MAX_AGENT_STEPS):
            undone_count = sum(1 for r in active_runs if not r["done"])
            if undone_count == 0:
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
    avg_steps = sum(len(t.steps) for t in all_trajectories) / max(total, 1)
    print(f"\n  Results: {successes}/{total} = {successes/total:.1%}")
    print(f"  Avg steps: {avg_steps:.1f}")

    return all_trajectories


def main():
    parser = argparse.ArgumentParser(description="Agent loop driver")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-gen-tokens", type=int, default=None)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument(
        "--retrieval-backend",
        default="heuristic",
        choices=["heuristic", "oracle"],
    )
    parser.add_argument(
        "--shuffle-mode", default=SHUFFLE_MODE_DEFAULT,
        choices=["per_question", "legacy"],
        help="per_question (v2, default) or legacy (the earlier version global pool)",
    )
    parser.add_argument(
        "--question-set", default="main", choices=["main", "dev"],
        help="main = historical 300; dev = fixed disjoint 100 for Gate 0",
    )
    parser.add_argument("--tag", default="", help="optional suffix for output file names")
    args = parser.parse_args()

    model_path = MODELS[args.model]
    print(f"Model: {args.model} ({model_path})")
    print(f"GPU: {args.gpu}")
    print(f"Retrieval backend: {args.retrieval_backend}")

    questions, shuffle_pool, data_meta = load_run_data(
        args.question_set, args.num_questions, args.shuffle_mode)
    out_dir = create_run_dir(f"{args.model}_pilot{('_'+args.tag) if args.tag else ''}", args, data_meta)
    llm = BatchLocalLLM(model_path, gpu_ids=str(args.gpu), thinking=args.thinking,
                        max_tokens=args.max_gen_tokens)
    print(f"Sampled {len(questions)} questions; output: {out_dir}")

    # Save questions
    q_file = out_dir / f"sampled_questions_{args.question_set}.json"
    with open(q_file, "x") as f:
        json.dump(
            [{"id": q["id"], "question": q["question"],
              "answer": q["answer"], "type": q["type"]}
             for q in questions],
            f, indent=2,
        )

    conditions = args.conditions or list(DEFAULT_PILOT_CONDITIONS)
    all_results = {}
    start = time.time()
    backend_suffix = "" if args.retrieval_backend == "heuristic" else f"_{args.retrieval_backend}"
    if args.question_set == "dev":
        backend_suffix += "_dev"
    if args.tag:
        backend_suffix += f"_{args.tag}"

    for cond in conditions:
        if cond not in FEEDBACK_CONDITIONS:
            print(f"Unknown condition: {cond}")
            continue

        t0 = time.time()
        trajectories = run_condition(
            llm, questions, cond, shuffle_pool, args.retrieval_backend,
            shuffle_mode=args.shuffle_mode,
        )
        elapsed = time.time() - t0

        result_data = [trajectory_to_dict(t, full=True) for t in trajectories]
        all_results[cond] = result_data

        out_file = out_dir / f"{args.model}_{cond}{backend_suffix}.json"
        with open(out_file, "x") as f:
            json.dump(result_data, f, indent=2)
        print(f"  Saved: {out_file} ({elapsed:.0f}s)")

    # Combined
    if conditions == list(DEFAULT_PILOT_CONDITIONS):
        combined_name = f"{args.model}_all{backend_suffix}.json"
    else:
        condition_tag = "_".join(conditions)
        if len(conditions) == 1:
            combined_name = f"{args.model}_{condition_tag}_only{backend_suffix}.json"
        else:
            combined_name = f"{args.model}_{condition_tag}{backend_suffix}.json"

    combined = out_dir / combined_name
    with open(combined, "x") as f:
        json.dump(all_results, f, indent=2)

    complete_run(out_dir, llm.generation_config)
    total = time.time() - start
    print(f"\nTotal: {total:.0f}s ({total/60:.1f}min)")
    print(f"Saved: {combined}")


if __name__ == "__main__":
    main()

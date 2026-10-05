"""Summarize a run: success per failure regime, mean6, tool calls, and the time-matched contrast Δ.

    python evaluate.py results_v2/<run dir>
    python evaluate.py results_v2/<run dir> --judgments results_v2/<replay dir>

Δ needs the agent's judgment of every observation. Conditions that use the side channel (enforced rule, combination,
decide) record it in <run dir>/gate2_log.json. For the others (unaided, permit, budget, ...), replay it first with
`python run_judge_replay.py --run-dir results_v2/<run dir>` and pass the replay directory with --judgments.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the toolkit, if it is not installed
from judged_useless import answer_rate_after_run, time_matched_contrast  # noqa: E402

FAIL = ["persistent", "recover_after_1", "recover_after_2", "recover_after_3", "late_onset_from_3"]
MEAN6 = FAIL + ["clean"]
ORDER = MEAN6 + ["plausible"]
NOT_REGIMES = {"run_manifest", "collection_summary", "gate2_log", "questions_private", "summary"}
MODE_TO_REGIME = {"shuffled": "persistent", "real": "clean"}   # run logs record the environment mode for two regimes


def load_run(run_dir: Path) -> dict:
    """question_id -> regime -> trajectory"""
    files = {p.stem: p for p in run_dir.glob("*.json") if p.stem not in NOT_REGIMES}
    data: dict = {}
    for reg in ORDER + sorted(set(files) - set(ORDER)):
        if reg in files:
            for t in json.loads(files[reg].read_text()):
                data.setdefault(t["question_id"], {})[reg] = t
    return data


def load_judgments(run_dir: Path, replay_dir: Path | None) -> dict:
    """(question_id, regime, t) -> judgment of the t-th observation"""
    out = {}
    if replay_dir is not None:
        for p in replay_dir.glob("*.json"):
            if p.stem in FAIL:
                for r in json.loads(p.read_text()):
                    out[(r["question_id"], p.stem, r["t"])] = r["judgment"]
    elif (run_dir / "gate2_log.json").exists():
        for x in json.loads((run_dir / "gate2_log.json").read_text()):
            reg = MODE_TO_REGIME.get(x["feedback_mode"], x["feedback_mode"])
            out[(x["question_id"], reg, x["t"])] = x["own_judgment"]
    return out


def tool_calls(t: dict) -> int:
    return sum(1 for s in t.get("steps", []) if s.get("action") in ("search", "lookup", "backup_search"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--judgments", help="replay directory written by run_judge_replay.py")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples for the interval of Δ")
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    data = load_run(run_dir)
    if not data:
        sys.exit(f"no trajectories found in {run_dir}")

    regimes = [r for r in ORDER if any(r in x for x in data.values())]
    regimes += sorted({r for x in data.values() for r in x} - set(regimes))
    print(f"{'regime':20s} {'n':>5s} {'success':>8s} {'tool calls':>11s}")
    for r in regimes:
        ts = [x[r] for x in data.values() if r in x]
        print(f"{r:20s} {len(ts):5d} {statistics.mean(str(t['success']) == 'True' for t in ts):8.3f} "
              f"{statistics.mean(tool_calls(t) for t in ts):11.2f}")
    full = [x for x in data.values() if all(r in x for r in MEAN6)]
    if full:
        mean6 = statistics.mean(sum(str(x[r]["success"]) == "True" for r in MEAN6) / 6 for x in full)
        print(f"\nmean6 (equal weight over the six regimes, {len(full)} questions): {mean6:.3f}")

    judg = load_judgments(run_dir, Path(a.judgments) if a.judgments else None)
    if not judg:
        print("\nNo judgments for this run, so Δ is not computed. Replay them with "
              f"`python run_judge_replay.py --run-dir {run_dir}` and pass --judgments <replay dir>.")
        return
    episodes = []
    for q, x in data.items():
        for r in FAIL:
            if r in x:
                n = len(x[r].get("observation_meta") or [])
                episodes.append({"question_id": q, "regime": r, "actions": [s["action"] for s in x[r]["steps"]],
                                 "judgments": [judg.get((q, r, t)) for t in range(1, n + 1)]})
    d = time_matched_contrast(episodes, n_boot=a.boot)
    print(f"Δ (own judgments, decisions after 3-6 observations): {d['delta']:+.3f} "
          f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]")
    p = answer_rate_after_run([e for e in episodes if e["regime"] == "persistent"], k=5)
    if p["reached"]:
        print(f"answers after five consecutive useless judgments (persistent): {p['answered']}/{p['reached']}")


if __name__ == "__main__":
    main()

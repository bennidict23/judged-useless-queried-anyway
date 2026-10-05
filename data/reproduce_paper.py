"""Recompute every Δ in Figure 3 and every mean6 in Table 2 of the paper from the released episodes (no GPU needed).

    python data/reproduce_paper.py            # a few minutes
    python data/reproduce_paper.py --boot 200 # faster, intervals then differ slightly from the paper
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # the toolkit, if it is not installed
from judged_useless import load_jsonl, time_matched_contrast  # noqa: E402

FAIL = ["persistent", "recover_after_1", "recover_after_2", "recover_after_3", "late_onset_from_3"]
MEAN6 = FAIL + ["clean"]
MODELS = ["qwen2.5-7b", "llama3.1-8b", "qwen3-8b", "qwen3-32b"]
CONDITIONS = ["unaided", "permit", "budget", "stated_rule", "call_cost", "decide", "enforced_rule", "combo"]


def mean6(episodes):
    by_q = {}
    for e in episodes:
        by_q.setdefault(e["question_id"], {})[e["regime"]] = e["success"]
    full = [x for x in by_q.values() if all(r in x for r in MEAN6)]
    return statistics.mean(sum(x[r] for r in MEAN6) / 6 for x in full)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    paper = json.loads((HERE / "episodes" / "paper_values.json").read_text())
    n_ok = n = 0
    for split in ["test300", "fresh300"]:
        print(f"\n{split}: Δ on the agent's own judgments [95% interval], and mean6 success")
        print(f"{'model':12s} {'condition':14s} {'Δ':>27s} {'mean6':>7s}  matches paper")
        for m in MODELS:
            for c in CONDITIONS:
                key = f"{split}/{m}__{c}"
                path = HERE / "episodes" / f"{key}.jsonl.gz"
                if not path.exists():
                    continue
                eps = load_jsonl(path)
                d = time_matched_contrast([e for e in eps if e["regime"] in FAIL], n_boot=a.boot)
                s = mean6(eps)
                p = paper[key]
                ok = d["delta"] == p["delta"] and s == p["mean6"] and (
                    a.boot != 2000 or (d["ci_low"], d["ci_high"]) == (p["ci_low"], p["ci_high"]))
                n += 1
                n_ok += ok
                print(f"{m:12s} {c:14s} {d['delta']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] {s:7.3f}  "
                      f"{'yes' if ok else 'NO'}")
    print(f"\n{n_ok} of {n} cells match the paper.")


if __name__ == "__main__":
    main()

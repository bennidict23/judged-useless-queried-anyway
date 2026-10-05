"""Δ and the answer rate after five useless judgments for the two example agents.

    python examples/compute_delta.py
"""
from pathlib import Path

from judged_useless import answer_rate_after_run, load_jsonl, time_matched_contrast

HERE = Path(__file__).parent

for name in ["unaided", "enforced_rule"]:
    episodes = load_jsonl(HERE / f"qwen3-8b_{name}_test300.jsonl")
    d = time_matched_contrast(episodes)
    persistent = [e for e in episodes if e["regime"] == "persistent"]
    a = answer_rate_after_run(persistent, k=5)
    print(f"Qwen3-8B, {name:13s}  Δ = {d['delta']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]   "
          f"answers after 5 useless judgments: {a['answered']}/{a['reached']}")

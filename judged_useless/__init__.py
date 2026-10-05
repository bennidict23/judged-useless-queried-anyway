"""Measure whether a tool-using agent stops on its own evidence judgments, and enforce the integration step.

    from judged_useless import load_jsonl, time_matched_contrast
    print(time_matched_contrast(load_jsonl("examples/qwen3-8b_unaided_test300.jsonl")))
"""
from .delta import answer_rate_after_run, decision_points, load_jsonl, time_matched_contrast, useless_runs
from .rule import (FORCE_SYSTEM_SUFFIX, FORCE_USER_MESSAGE, SIDE_CHANNEL_QUESTION, IntegrationRule,
                   parse_side_channel, stated_judgment)

__all__ = ["time_matched_contrast", "answer_rate_after_run", "decision_points", "useless_runs", "load_jsonl",
           "IntegrationRule", "stated_judgment", "parse_side_channel", "SIDE_CHANNEL_QUESTION",
           "FORCE_SYSTEM_SUFFIX", "FORCE_USER_MESSAGE"]
__version__ = "1.0.0"

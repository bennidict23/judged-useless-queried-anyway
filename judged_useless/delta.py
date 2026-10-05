"""Stopping measures from an agent's own per-step judgments.

Input format: one trajectory per record (a dict), with

    question_id   cluster for the bootstrap (trajectories of one question in several regimes form one cluster)
    actions       the action at each step, e.g. ["search", "search", "lookup", "finish"]
    judgments     the agent's judgment of each observation, in order: "USELESS", "USEFUL" or None
                  (judgments[k] is the judgment of the observation returned by actions[k])

A decision point is every action taken after at least one observation: action i (0-based, i >= 1) is taken after
observations 1..i. The agent "answers" at i if actions[i] == "finish".

time_matched_contrast() is the paper's Delta: at the same decision t, the probability of answering when every
observation so far was judged useless, minus the probability of answering when the latest observation was judged
useless but an earlier one was judged useful. It is positive for an agent whose stopping integrates its own useless
judgments, zero for an agent that stops by the clock or at a deadline, and negative when stopping follows earlier
useful evidence. Strata are pooled with the Mantel-Haenszel risk difference; intervals come from a bootstrap over
clusters (questions). No dependencies beyond the standard library.
"""
from __future__ import annotations

import collections
import gzip
import json
import random
from typing import Iterable, Iterator, Mapping, Sequence

USELESS = "USELESS"


def load_jsonl(path) -> list[dict]:
    """Read trajectories from a JSON-lines file (plain or .gz)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def useless_runs(judgments: Sequence[str | None], n: int | None = None) -> dict[int, int]:
    """{t: length of the run of consecutive USELESS judgments that ends at observation t} for t = 1..n."""
    n = len(judgments) if n is None else n
    run, out = 0, {}
    for t in range(1, n + 1):
        j = judgments[t - 1] if t <= len(judgments) else None
        run = run + 1 if j == USELESS else 0
        out[t] = run
    return out


def decision_points(trajectory: Mapping) -> Iterator[tuple[int, int, bool]]:
    """Yield (t, run, answered) for each decision taken after t >= 1 observations, where run is the number of
    consecutive USELESS judgments ending at observation t."""
    actions, judgments = trajectory["actions"], trajectory["judgments"]
    last = min(len(actions) - 1, len(judgments))
    runs = useless_runs(judgments, last)
    for t in range(1, last + 1):
        yield t, runs[t], actions[t] == "finish"


def _clusters(trajectories: Iterable[Mapping], cluster_key: str) -> list[list[Mapping]]:
    groups: dict = {}
    for tr in trajectories:
        groups.setdefault(tr[cluster_key], []).append(tr)
    return list(groups.values())


def _mh(rows: list[dict]) -> tuple[float, dict]:
    tot = collections.defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        for t, v in r.items():
            for k in range(4):
                tot[t][k] += v[k]
    num = den = 0.0
    for t, (a1, n1, a0, n0) in tot.items():
        if n1 and n0:
            w = n1 * n0 / (n1 + n0)
            num += w * (a1 / n1 - a0 / n0)
            den += w
    return (num / den if den else float("nan")), {int(t): list(v) for t, v in sorted(tot.items())}


def time_matched_contrast(trajectories: Iterable[Mapping], t_min: int = 3, t_max: int = 6, n_boot: int = 2000,
                          seed: int = 0, cluster_key: str = "question_id") -> dict:
    """Delta with a percentile bootstrap interval over clusters.

    t_min, t_max: decision points pooled (after t_min .. t_max observations). The paper uses 3..6 with an
    eight-action budget, i.e. it excludes the final action, which is the last chance to answer before the deadline.

    Returns {"delta", "ci_low", "ci_high", "cells", "n_clusters"}, where cells[t] =
    [answered, n] for "all observations so far judged useless" followed by [answered, n] for "latest useless,
    an earlier one useful".
    """
    per_cluster = []
    for group in _clusters(trajectories, cluster_key):
        c = collections.defaultdict(lambda: [0, 0, 0, 0])
        for tr in group:
            for t, run, answered in decision_points(tr):
                if t < t_min or t > t_max:
                    continue
                if run == t:                      # every observation so far judged useless
                    c[t][0] += answered
                    c[t][1] += 1
                elif 1 <= run <= t - 2:           # latest useless, an earlier one useful
                    c[t][2] += answered
                    c[t][3] += 1
        per_cluster.append(dict(c))

    delta, cells = _mh(per_cluster)
    rng = random.Random(seed)
    boots = [_mh([per_cluster[rng.randrange(len(per_cluster))] for _ in per_cluster])[0] for _ in range(n_boot)]
    boots = sorted(b for b in boots if b == b)
    lo = boots[int(0.025 * len(boots))] if boots else float("nan")
    hi = boots[int(0.975 * len(boots)) - 1] if boots else float("nan")
    return {"delta": delta, "ci_low": lo, "ci_high": hi, "cells": cells, "n_clusters": len(per_cluster)}


def answer_rate_after_run(trajectories: Iterable[Mapping], k: int = 5) -> dict:
    """Among trajectories whose judgments reach k consecutive USELESS at a decision point (i.e. with an action still
    to take), the share in which the agent answers (its final action is finish) rather than running out of budget."""
    reached = answered = 0
    for tr in trajectories:
        last = min(len(tr["actions"]) - 1, len(tr["judgments"]))
        if max(useless_runs(tr["judgments"], last).values(), default=0) >= k:
            reached += 1
            answered += bool(tr["actions"]) and tr["actions"][-1] == "finish"
    return {"reached": reached, "answered": answered, "rate": answered / reached if reached else float("nan")}

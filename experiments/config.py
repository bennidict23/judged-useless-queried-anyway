"""Experiment configuration: paths, models, decoding and sampling settings."""

import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# v2: new runs go here; results/ is the frozen archive of the earlier version.
RESULTS_DIR_V2 = PROJECT_ROOT / "results_v2"
RESULTS_DIR_V2.mkdir(exist_ok=True)
SAMPLING_VERSION = "v2.1-manifest"


def stable_seed(*parts) -> int:
    """Process-independent integer seed derived from arbitrary parts.

    Replaces `SEED + hash(condition)`, which depended on PYTHONHASHSEED.
    """
    h = hashlib.sha256("|".join(str(x) for x in parts).encode("utf-8")).hexdigest()
    return int(h[:15], 16)

MODELS = {
    "qwen3-8b": "models/Qwen3-8B",
    "qwen3-32b": "models/Qwen3-32B",   # local, tensor-parallel over 4 GPUs (--gpu 0,1,2,3)
    "llama3.1-8b": "models/Llama-3.1-8B-Instruct",
    "qwen2.5-7b": "models/Qwen2.5-7B-Instruct",
    # Official Anthropic API (prefix "anthropic:"); key via ANTHROPIC_API_KEY / ANTHROPIC_API_KEY_FILE
    "anthropic-sonnet-5": "anthropic:claude-sonnet-5",
    "anthropic-haiku-4.5": "anthropic:claude-haiku-4-5-20251001",
}

# ── Feedback conditions ──
# All use the EXACT same system prompt. Only difference: what observation
# the agent receives after each action.
FEEDBACK_CONDITIONS = {
    "real": "Agent receives actual, correct observations from the environment.",
    "shuffled": "Agent receives real paragraphs, but from unrelated questions.",
    "plausible": (
        "Agent receives real-looking distractor pages from the same question "
        "instead of the supporting pages."
    ),
    "conflicting": (
        "Agent first receives a same-question distractor page, then later "
        "searches recover real evidence again."
    ),
    "no_feedback": "Agent receives 'no results found' and must use parametric knowledge.",
}
DEFAULT_PILOT_CONDITIONS = ["real", "shuffled", "no_feedback"]

# ── Shuffled-observation sampling ──
#   "legacy":       global pool, every environment reads from index 0 -> all
#                   questions see the same first <=8 paragraphs (the earlier version).
#   "per_question": paragraphs sampled without replacement from OTHER questions'
#                   pages with a seed derived from (SHUFFLE_SALT, question_id).
#                   With the same pool, persistent and recover-after-k share
#                   the first k misleading observations, not all histories.
SHUFFLE_MODE_DEFAULT = "per_question"
SHUFFLE_SALT = "shuffled-v2"
# Replicate runs (addendum 10): REPLICATE=<tag> draws a new misleading-paragraph order and new sampling seeds while
# questions, regimes and prompts stay fixed. Unset -> the original runs' seeds, bit for bit.
REPLICATE = __import__('os').environ.get('REPLICATE', '')
if REPLICATE:
    SHUFFLE_SALT = f"{SHUFFLE_SALT}-rep{REPLICATE}"

# ── F protocol: "answer now without further tool evidence" ──
# Accepted. Both stop tools without claiming failure or naming
# a knowledge source. Legacy no_feedback is a distinct strategy intervention.
F_PROTOCOL_MAIN = (
    "Do not make any further tool calls. Give your final answer to the original "
    "question using finish[answer]."
)
# STRICT (sensitivity arm): same user-slot text as MAIN plus a system-slot
# statement that finish is the only available action this turn. Measures how much of MAIN's
# non-compliance is instruction-following rather than reluctance to stop.
F_PROTOCOL_STRICT_SYSTEM_SUFFIX = (
    "\n\nFor this turn only, the search and lookup actions are unavailable; the only available action "
    "is finish[answer]."
)
F_PROTOCOL_ALT = (
    "Answer the original question now, without making additional tool calls. "
    "Use finish[answer] for your final answer."
)
F_PROTOCOL_LEGACY = (
    "No relevant results found for your query. Please reason based on what you "
    "already know and give your final answer using finish[answer]."
)

# ── Dev / validation split ──
# Original 300 = historical replication, NOT untouched validation. Dev 100:
# first 20 for rubric development, remaining 80 reserved for rubric audit.
NUM_DEV_QUESTIONS = 100
NUM_RUBRIC_QUESTIONS = 20
DEV_SEED_TAG = "dev-v2"

# ── Agent parameters ──
# Step budget. Default 8 for every reported run; the budget-scaling addendum overrides it via the
# MAX_AGENT_STEPS environment variable, and run_gate2.py records the value in the run manifest.
MAX_AGENT_STEPS = int(__import__('os').environ.get('MAX_AGENT_STEPS', '8'))
MAX_GEN_TOKENS = 400       # same budget for all conditions
NUM_QUESTIONS = 300
BATCH_SIZE = 32

# ── Inference ──
VLLM_TENSOR_PARALLEL = 1
TEMPERATURE = 0.0
SEED = 42

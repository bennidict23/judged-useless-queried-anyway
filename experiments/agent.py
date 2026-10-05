"""ReAct agent — same prompt for all conditions."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# CRITICAL: The SAME system prompt for all feedback conditions.
# No confound — only the observation content differs.
SYSTEM_PROMPT = """\
You are a question-answering agent. You answer questions by interacting with a Wikipedia-like knowledge base.

Available actions:
- search[query]: Search for a Wikipedia article. Returns the article text.
- lookup[keyword]: Find the next sentence containing the keyword in the current article.
- finish[answer]: Submit your final answer. The answer should be short (a few words).

Format your response as:
Thought: <your step-by-step reasoning>
Action: <action_name>
Action Input: <input>"""

# v2 protocol makes the existing one-tool-call interaction boundary explicit.
# Applied identically in every regime, while legacy runs retain the old prompt.
SYSTEM_PROMPT_V2 = SYSTEM_PROMPT + """

Produce exactly one Thought and one Action per turn. End your response after
Action Input and wait for the tool observation. Do not simulate tool results
or future actions."""


PLAN_SYSTEM_PROMPT = """\
You are planning how to answer a multi-hop factual question using a Wikipedia-like knowledge base.

Before seeing any search results, write a short execution plan:
- which entity or phrase to search first
- what second clue or bridge fact you expect to need
- what kind of final answer you expect to extract

Keep the plan short and concrete.
Format:
Plan: <2-4 short sentences>"""


PLAN_THEN_ACT_SYSTEM_PROMPT = """\
You are a question-answering agent. You answer questions by interacting with a Wikipedia-like knowledge base.

You have already made an initial plan before seeing any evidence.
Use that plan to guide your first action, but update it if later observations contradict it.
Do not blindly trust any single observation.

Available actions:
- search[query]: Search for a Wikipedia article. Returns the article text.
- lookup[keyword]: Find the next sentence containing the keyword in the current article.
- finish[answer]: Submit your final answer. The answer should be short (a few words).

Format your response as:
Thought: <brief reasoning that refers to the current evidence and your plan>
Action: <action_name>
Action Input: <input>"""


@dataclass
class AgentStep:
    thought: str
    action: str
    action_input: str
    observation: str
    thought_tokens: int = 0
    raw_response: str = ""          # v2: untruncated model output for this step


@dataclass
class AgentTrajectory:
    question_id: str
    question: str
    gold_answer: str
    feedback_mode: str
    initial_plan: str | None = None
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str | None = None
    success: bool = False
    num_searches: int = 0
    num_lookups: int = 0
    used_parametric: bool = False  # did agent answer without using observations?
    # v2 fields: full logging for replay and decision-point analysis
    termination_reason: str | None = None   # answered | budget_exhausted | format_failure
    messages: list[dict] | None = None      # full chat context at termination
    observation_meta: list[dict] = field(default_factory=list)  # env design metadata per observation
    shuffled_sources: list[dict] = field(default_factory=list)  # provenance of misleading paragraphs
    sampling_version: str = "v1"
    shuffle_mode: str = "legacy"
    generation_config: dict = field(default_factory=dict)
    generations: list[dict] = field(default_factory=list)


def parse_agent_response(response: str, strict: bool = False) -> tuple[str, str, str]:
    # Only the response after Qwen's thinking block specifies an action.
    # Raw output is preserved separately. An unclosed block is incomplete.
    if "</think>" in response:
        response = response.split("</think>", 1)[1].strip()
    elif "<think>" in response:
        return "", "", ""
    if strict and len(re.findall(r'^\s*Action\s*:', response, re.IGNORECASE | re.MULTILINE)) > 1:
        return "", "", ""
    thought, action, action_input = "", "", ""

    standalone = re.fullmatch(r'(search|lookup|finish)\[(.+)\]', response.strip(), re.IGNORECASE)
    if standalone:
        return "", standalone.group(1).lower(), standalone.group(2)

    action_match = re.search(r'Action\s*:\s*(.+)', response, re.IGNORECASE)
    if action_match:
        thought = response[:action_match.start()].strip()
        thought = re.sub(r'^Thought\s*:\s*', '', thought, flags=re.IGNORECASE).strip()
        action = action_match.group(1).strip()

    input_match = re.search(r'Action\s*Input\s*:\s*(.+)', response, re.IGNORECASE)
    if input_match:
        action_input = input_match.group(1).strip()
    else:
        bracket_match = re.search(r'(\w+)\[(.+?)\]', action)
        if bracket_match:
            action = bracket_match.group(1)
            action_input = bracket_match.group(2)

    action = re.sub(r'\[.*\]', '', action).strip().lower()
    if not action and os.environ.get('AGENT_PARSE_XML_TOOLCALLS') == '1':
        # Amendment 20 (Claude Haiku 4.5 prompt arms only): the model sometimes emits its native tool-call markup
        # instead of an Action line; read the first call as the action. Off by default, so earlier runs are unchanged.
        xml = re.search(r'<invoke name="(search|lookup|finish)">\s*<parameter name="[^"]*">(.*?)</parameter>',
                        response, re.IGNORECASE | re.DOTALL)
        if xml:
            thought = re.sub(r'^Thought\s*:\s*', '', response[:response.find('<function_calls>')].strip() if '<function_calls>' in response
                             else response[:xml.start()].strip(), flags=re.IGNORECASE).strip()
            return thought, xml.group(1).lower(), xml.group(2).strip()
    return thought, action, action_input


def normalize_answer(answer: str) -> str:
    answer = answer.lower().strip()
    answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
    answer = re.sub(r'[^\w\s]', '', answer)
    answer = re.sub(r'\s+', ' ', answer).strip()
    return answer


def check_answer(predicted: str, gold: str) -> bool:
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    if pred_norm == gold_norm:
        return True

    pred_tokens = set(pred_norm.split())
    gold_tokens = set(gold_norm.split())
    if not pred_tokens or not gold_tokens:
        return False

    common = pred_tokens & gold_tokens
    if not common:
        return False

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1 >= 0.6


def trajectory_to_dict(t: AgentTrajectory, full: bool = True) -> dict:
    """Serialize a trajectory.

    full=True (v2 default): untruncated thought/observation plus raw responses,
    termination reason, full messages and environment metadata.
    full=False reproduces the earlier-version archive format (thought[:300],
    observation[:200]).
    """
    if not full:
        return _trajectory_to_dict_legacy(t)
    return {
        "question_id": t.question_id,
        "question": t.question,
        "gold_answer": t.gold_answer,
        "feedback_mode": t.feedback_mode,
        "initial_plan": t.initial_plan,
        "final_answer": t.final_answer,
        "success": t.success,
        "num_steps": len(t.steps),
        "num_searches": t.num_searches,
        "num_lookups": t.num_lookups,
        "termination_reason": t.termination_reason,
        "sampling_version": t.sampling_version,
        "shuffle_mode": t.shuffle_mode,
        "generation_config": t.generation_config,
        "generations": t.generations,
        "steps": [
            {
                "thought": s.thought,
                "action": s.action,
                "action_input": s.action_input,
                "observation": s.observation,
                "raw_response": s.raw_response,
            }
            for s in t.steps
        ],
        "observation_meta": t.observation_meta,
        "shuffled_sources": t.shuffled_sources,
        "messages": t.messages,
    }


def _trajectory_to_dict_legacy(t: AgentTrajectory) -> dict:
    return {
        "question_id": t.question_id,
        "question": t.question,
        "gold_answer": t.gold_answer,
        "feedback_mode": t.feedback_mode,
        "initial_plan": t.initial_plan,
        "final_answer": t.final_answer,
        "success": t.success,
        "num_steps": len(t.steps),
        "num_searches": t.num_searches,
        "num_lookups": t.num_lookups,
        "steps": [
            {
                "thought": s.thought[:300],
                "action": s.action,
                "action_input": s.action_input,
                "observation": s.observation[:200],
            }
            for s in t.steps
        ],
    }

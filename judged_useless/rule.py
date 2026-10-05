"""The enforced integration step and the two ways to obtain the judgments it needs.

IntegrationRule turns an agent's per-observation judgments into a stopping decision: once k consecutive
observations are judged useless, the harness allows only the answer action. The paper uses k = 5.

Judgments can come from
  * a side channel: after each observation, ask SIDE_CHANNEL_QUESTION on a scratch copy of the conversation that
    the agent never sees again, and read the reply with parse_side_channel();
  * the agent's own reasoning: read the thought the agent writes after the observation with stated_judgment(), a
    keyword reader that needs no extra model call.

When the rule fires, append FORCE_SYSTEM_SUFFIX to the system message and FORCE_USER_MESSAGE to the latest
observation, and regenerate the turn.
"""
from __future__ import annotations

import re

SIDE_CHANNEL_QUESTION = ("Before acting: was the LAST observation useful for answering the original question? "
                         "Reply with exactly one word: USEFUL or USELESS.")
FORCE_SYSTEM_SUFFIX = ("\n\nFor this turn only, the search and lookup actions are unavailable; the only available "
                       "action is finish[answer].")
FORCE_USER_MESSAGE = "Do not make any further tool calls. Give your final answer to the original question using finish[answer]."


def parse_side_channel(reply: str) -> str:
    """USELESS, USEFUL or UNPARSED from the reply to SIDE_CHANNEL_QUESTION."""
    t = reply.upper()
    if "USELESS" in t:
        return "USELESS"
    if "USEFUL" in t:
        return "USEFUL"
    return "UNPARSED"


# Keyword reader for the judgment an agent states in its own reasoning. Frozen before the held-out runs; only the
# first two sentences of the thought are read, where agents comment on the observation they just saw.
_NEG = [
    r'\bnot (?:directly |really |very |at all |)(?:relevant|related|helpful|useful|pertinent)\b',
    r'\b(?:irrelevant|unrelated)\b',
    r'\b(?:does|do|did)(?: not|n\'t) (?:seem to |appear to |)(?:explicitly |directly |specifically |clearly |actually |)(?:mention|provide|contain|include|give|answer|say|show|have|'
    r'state|specify|address|discuss|relate|help|offer|list|indicate|confirm|match|reveal|return|yield|lead|pertain)\b',
    r'\b(?:is|are|was|were)(?: still|) not (?:about|relevant|related|helpful|useful|mentioned|provided|found|available|'
    r'present|listed|given|what)\b',
    r'\b(?:isn\'t|aren\'t|wasn\'t|weren\'t) (?:about|relevant|related|helpful|useful|mentioned|what)\b',
    r'\bno (?:relevant |useful |direct |specific |)(?:information|mention|details?|results?|data|answer|indication)\b',
    r'\bnothing (?:about|on|related|relevant|useful)\b',
    r'\b(?:still|again) (?:not|no|nothing)\b',
    r'\b(?:unable|failed) to (?:find|locate|get)\b',
    r'\b(?:has|have|had)(?: not|n\'t)(?: yet| still|) (?:provided|returned|yielded|given|mentioned|found|shown|revealed|helped)\b',
    r'\b(?:refer|refers|referred|referring|relate|relates|is about|are about|describes?|focus(?:es)? on|point to)\b[^.;]{0,90}?,? (?:and |but |)not (?:the |a |an |to |about |)\w',
    r'\b(?:instead of|rather than) (?:the |a |an |)\w+',
    r'\b(?:could|can)(?: not|n\'t) find\b',
    r'\bnot (?:returning|finding|giving|providing|showing)\b',
    r'\bnot what (?:i|we) (?:was|were|am|are) looking for\b',
    r'\b(?:wrong|different|another|unrelated) (?:article|page|topic|person|entity|subject)\b',
    r'\b(?:off[- ]topic|not on topic)\b',
    r'\blacks? (?:any |the |)(?:information|mention|details?)\b',
]
_POS = [
    r'\b(?:the )?answer (?:is|to the question is)\b',
    r'\b(?:this|that|it) (?:answers|confirms)\b',
    r'\bconfirms? that\b',
    r'\b(?:provides?|gives?|states?|mentions?|shows?|indicates?|reveals?|lists?|says?) (?:that |the |a |an |his |her |their |its |)'
    r'(?!no\b)',
    r'\bnow (?:i|we) (?:know|have)\b',
    r'\baccording to (?:the|this)\b',
]
_NEG_RE = [re.compile(p, re.I) for p in _NEG]
_POS_RE = [re.compile(p, re.I) for p in _POS]
_SPLIT = re.compile(r'(?<=[.!?])\s+')


def stated_judgment(thought: str) -> str:
    """USELESS if the first two sentences of the thought say the result does not help; USEFUL if they report
    something found and do not say it does not help; NONE otherwise."""
    head = " ".join(_SPLIT.split((thought or "").strip())[:2])
    if any(p.search(head) for p in _NEG_RE):
        return "USELESS"
    if any(p.search(head) for p in _POS_RE):
        return "USEFUL"
    return "NONE"


class IntegrationRule:
    """Answer after k consecutive observations judged useless.

    source="side_channel": any judgment other than USELESS resets the run (as with the side-channel question).
    source="stated": USELESS extends the run, USEFUL resets it, NONE (no explicit judgment) leaves it unchanged.
    Once the rule has fired it stays fired, so only the answer action remains for the rest of the episode.
    """

    def __init__(self, k: int = 5, source: str = "side_channel"):
        if source not in ("side_channel", "stated"):
            raise ValueError("source must be 'side_channel' or 'stated'")
        self.k, self.source = k, source
        self.run, self.fired = 0, False

    def reset(self) -> None:
        self.run, self.fired = 0, False

    def update(self, judgment: str) -> bool:
        """Record the judgment of the latest observation; return True if the next action must be the answer."""
        if judgment == "USELESS":
            self.run += 1
        elif self.source == "side_channel" or judgment == "USEFUL":
            self.run = 0
        self.fired = self.fired or self.run >= self.k
        return self.fired

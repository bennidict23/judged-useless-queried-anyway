"""Zero-cost reader of the judgment an agent states in its own reasoning (amendment 19).

Reads the Thought the agent writes right after an observation and returns USELESS (it says the result does not help),
USEFUL (it reports finding something it can use) or NONE (no explicit judgment). Keyword rules only, developed on the
development-set stated-judgment labels (rubric v0.3.2, stream S2) and frozen before any held-out run; no model call.
Only the first two sentences are read, where agents comment on the observation they just saw.
"""
from __future__ import annotations
import re

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
NEG = [re.compile(p, re.I) for p in _NEG]
POS = [re.compile(p, re.I) for p in _POS]
_SPLIT = re.compile(r'(?<=[.!?])\s+')


def head(thought: str, n: int = 2) -> str:
    return ' '.join(_SPLIT.split((thought or '').strip())[:n])


def stated_judgment(thought: str) -> str:
    """USELESS if the first two sentences say the result does not help; USEFUL if they report something found and do
    not say it does not help; NONE otherwise."""
    h = head(thought)
    if any(p.search(h) for p in NEG):
        return 'USELESS'
    if any(p.search(h) for p in POS):
        return 'USEFUL'
    return 'NONE'

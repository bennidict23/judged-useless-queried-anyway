"""Cheap non-LLM usefulness detector (baseline signal for Gate 2).

USELESS if the latest observation shares too few content words with the question: a proxy for a
source-reliability tracker that only sees surface overlap (the kind of signal a tool-memory / reliability
baseline could use without any model judgment). 'No results' lookups are USELESS.
"""
from __future__ import annotations
import re

STOP = set('''a an the of in on at to for from by with and or but is are was were be been being which who whom whose what when
where why how that this these those it its as into than then there their they he she his her him them do does did not no
nor so such both either neither one two first second also about after before during under over between same other
more most many much any some each every all only own very can could would should may might will shall has have had
name named called known american film series song band album born year city town county state country'''.split())


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOP}


def lexical_useless(question: str, observation: str, tau: float = 0.25) -> bool:  # tau chosen on dev (audit80 final S1 labels) by BALANCED accuracy 0.772 (useless recall 0.64, useful recall 0.91)
    obs = observation.split('Observation:', 1)[-1].strip()
    if obs.lower().startswith(('no results', 'no relevant results', 'no matching content')):
        return True
    q = content_words(question)
    if not q:
        return False
    head = obs[:400]  # title + first sentences, as a tracker would see
    return len(q & content_words(head)) / len(q) < tau

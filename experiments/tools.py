"""Environment with configurable feedback fidelity."""

from __future__ import annotations

import random
import re
from difflib import SequenceMatcher

from config import SHUFFLE_MODE_DEFAULT, SHUFFLE_SALT, stable_seed


class HotpotQAEnvironment:
    """Provides search/lookup/finish tools with configurable feedback mode.

    feedback_mode:
      - "real": correct observations (normal)
      - "shuffled": observations drawn from unrelated articles (shuffle_pool)
      - "plausible": observations drawn from same-question distractor pages
      - "conflicting": first search shows a same-question distractor, later
        searches/lookup return real evidence again
      - "no_feedback": generic "no results" message
      - "partial": XX% of observations are shuffled, rest are real
      - "targeted": corrupt only specific step numbers (corrupt_steps set)
      - "verified": like partial, but each observation is independently sampled
        twice; if the two samples disagree, return "no results" instead.
    """

    def __init__(
        self,
        knowledge_base: dict[str, str],
        feedback_mode: str = "real",
        shuffle_pool: list[str] | None = None,
        rng: random.Random | None = None,
        corruption_rate: float = 0.0,
        corrupt_steps: set[int] | None = None,
        retrieval_backend: str = "heuristic",
        oracle_titles: list[str] | None = None,
        distractor_titles: list[str] | None = None,
        question_id: str | None = None,
        shuffle_mode: str = "legacy",
        shuffle_salt: str = SHUFFLE_SALT,
    ):
        """
        shuffle_mode:
          - "legacy": `shuffle_pool` is a list[str]; every environment reads the
            global pool from index 0 (the earlier version behaviour, kept for reproduction).
          - "per_question": `shuffle_pool` is a list[dict] from
            data.build_shuffle_pool_v2; paragraphs are drawn without replacement
            from OTHER questions' pages with seed stable_seed(shuffle_salt,
            question_id). The sequence depends only on the question, so all
            regimes for the same question share identical misleading prefixes.
        """
        self.kb = knowledge_base
        self.mode = feedback_mode
        self.shuffle_mode = shuffle_mode
        self.question_id = question_id
        self.shuffle_pool = shuffle_pool or []
        self.shuffle_idx = 0
        self.shuffled_sources: list[dict] = []   # provenance of each served misleading paragraph
        self.observation_meta: list[dict] = []   # one entry per non-finish action (design metadata)
        if shuffle_mode == "per_question":
            if question_id is None:
                raise ValueError("per_question shuffle_mode requires question_id")
            if self.shuffle_pool and not isinstance(self.shuffle_pool[0], dict):
                raise TypeError("per_question shuffle_mode requires build_shuffle_pool_v2 entries")
            own_titles = {title.casefold().strip() for title in knowledge_base}
            candidates = [i for i, e in enumerate(self.shuffle_pool)
                          if e["source_qid"] != question_id
                          and e["title"].casefold().strip() not in own_titles]
            order_rng = random.Random(stable_seed(shuffle_salt, question_id))
            order_rng.shuffle(candidates)
            self._shuffle_order = candidates
        elif shuffle_mode != "legacy":
            raise ValueError(f"unknown shuffle_mode: {shuffle_mode}")
        self.rng = rng or random.Random(42)
        self.corruption_rate = corruption_rate
        self.corrupt_steps = corrupt_steps or set()
        self.retrieval_backend = retrieval_backend
        self.oracle_titles = [
            title for title in (oracle_titles or [])
            if title in self.kb
        ]
        self.distractor_titles = [
            title for title in (distractor_titles or [])
            if title in self.kb
        ]
        self.corruption_log: list[bool] = []  # True if this step was corrupted
        self.verification_log: list[dict] = []  # For verified mode: tracks each check
        self._step_counter = 0  # counts non-finish actions
        self._conflict_seeded = False  # whether the first distractor search was injected

        self.current_page: str | None = None
        self.current_title: str | None = None
        self.finished = False
        self.final_answer: str | None = None

        # Track what the agent tried (for analysis)
        self.action_log: list[dict] = []

    def execute(self, action: str, action_input: str) -> str:
        action = action.strip().lower()
        action_input = action_input.strip()

        if action == "finish":
            return self._finish(action_input)

        # For search/lookup, get the REAL result first (for logging),
        # then return according to feedback mode
        real_result = self._get_real_result(action, action_input)

        self.action_log.append({
            "action": action,
            "input": action_input,
            "real_result": real_result[:100] if real_result else "",
            "mode": self.mode,
            "retrieval_backend": self.retrieval_backend,
        })

        self._step_counter += 1
        n_shuffled_before = len(self.shuffled_sources)
        observation = self._dispatch(action, action_input, real_result)
        self.observation_meta.append({
            "obs_index": self._step_counter,            # 1-indexed count of tool observations
            "action": action,
            "regime_label": self._regime_label(observation, real_result, n_shuffled_before),
            "corrupted": observation != real_result,
        })
        return observation

    def _regime_label(self, observation: str, real_result: str, n_shuffled_before: int) -> str:
        """Design metadata only: what the environment did. NOT a semantic usefulness label."""
        if len(self.shuffled_sources) > n_shuffled_before:
            return "shuffled"
        if observation == real_result:
            return "real"
        if observation.startswith("No relevant results found") or observation.startswith("No matching content found"):
            return "no_feedback"
        return "distractor_or_other"

    def _dispatch(self, action: str, action_input: str, real_result: str) -> str:
        if self.mode == "real":
            return real_result
        elif self.mode == "shuffled":
            return self._get_shuffled_result()
        elif self.mode == "plausible":
            return self._get_plausible_result(action, action_input)
        elif self.mode == "conflicting":
            return self._get_conflicting_result(action, action_input, real_result)
        elif self.mode == "no_feedback":
            return self._get_no_feedback_result(action)
        elif self.mode == "partial":
            # Each observation independently corrupted with corruption_rate
            if self.rng.random() < self.corruption_rate:
                self.corruption_log.append(True)
                return self._get_shuffled_result()
            else:
                self.corruption_log.append(False)
                return real_result
        elif self.mode == "targeted":
            # Corrupt only specific step numbers (1-indexed)
            if self._step_counter in self.corrupt_steps:
                self.corruption_log.append(True)
                return self._get_shuffled_result()
            else:
                self.corruption_log.append(False)
                return real_result
        elif self.mode == "verified":
            # Re-query verification: sample observation twice independently,
            # accept only if both agree (word overlap > threshold).
            obs1 = self._sample_with_corruption(real_result)
            obs2 = self._sample_with_corruption(real_result)
            similarity = self._word_overlap(obs1, obs2)
            consistent = similarity >= 0.5

            self.verification_log.append({
                "consistent": consistent,
                "similarity": similarity,
                "obs1_corrupted": obs1 != real_result,
                "obs2_corrupted": obs2 != real_result,
            })

            if consistent:
                self.corruption_log.append(obs1 != real_result)
                return obs1
            else:
                # Inconsistent → at least one is corrupted → discard both
                self.corruption_log.append(False)  # not passing corruption through
                return self._get_no_feedback_result(action)
        else:
            return real_result

    def _sample_with_corruption(self, real_result: str) -> str:
        """Independently decide whether to corrupt this observation."""
        if self.rng.random() < self.corruption_rate:
            return self._get_shuffled_result()
        return real_result

    @staticmethod
    def _word_overlap(text1: str, text2: str) -> float:
        """Word-level Jaccard similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    def _get_real_result(self, action: str, action_input: str) -> str:
        if action == "search":
            return self._search_real(action_input)
        elif action == "lookup":
            return self._lookup_real(action_input)
        return f"Invalid action '{action}'. Valid: search, lookup, finish."

    def _search_real(self, query: str) -> str:
        if self.retrieval_backend == "oracle" and self.oracle_titles:
            return self._search_in_titles(query, self.oracle_titles, force_result=True)
        return self._search_in_titles(query, list(self.kb.keys()), force_result=False)

    def _search_in_titles(
        self,
        query: str,
        candidate_titles: list[str],
        force_result: bool,
    ) -> str:
        query_lower = query.lower()

        # Exact title match
        for title in candidate_titles:
            if query_lower == title.lower():
                return self._load_page(title)

        # Fuzzy title match
        best_title, best_score = None, 0.0
        for title in candidate_titles:
            score = SequenceMatcher(None, query_lower, title.lower()).ratio()
            if score > best_score:
                best_score = score
                best_title = title

        if best_title and best_score > 0.3:
            return self._load_page(best_title)

        # Content fallback
        best_title, best_overlap = None, 0
        query_words = set(query_lower.split())
        for title in candidate_titles:
            paragraph = self.kb[title]
            overlap = len(query_words & set(paragraph.lower().split()))
            if overlap > best_overlap:
                best_overlap = overlap
                best_title = title

        if best_title:
            return self._load_page(best_title)

        if force_result and candidate_titles:
            # Oracle retrieval should always expose a relevant page if one exists.
            return self._load_page(sorted(candidate_titles)[0])

        return "No results found."

    def _load_page(self, title: str) -> str:
        paragraph = self.kb[title]
        self.current_page = paragraph
        self.current_title = title
        return f"[{title}] {paragraph}"

    def _lookup_real(self, keyword: str) -> str:
        if self.current_page is None:
            return "No page loaded. Use search first."
        keyword_lower = keyword.lower()
        sentences = re.split(r'(?<=[.!?])\s+', self.current_page)
        for i, sent in enumerate(sentences):
            if keyword_lower in sent.lower():
                return f"(Result {i+1}/{len(sentences)}) {sent}"
        return f"No results for '{keyword}' in current page."

    def _get_shuffled_result(self) -> str:
        """Return a real-looking but irrelevant paragraph from the pool."""
        if not self.shuffle_pool:
            return "No results found."
        if self.shuffle_mode == "per_question":
            if not self._shuffle_order:
                return "No results found."
            pool_idx = self._shuffle_order[self.shuffle_idx % len(self._shuffle_order)]
            entry = self.shuffle_pool[pool_idx]
            self.shuffle_idx += 1
            self.shuffled_sources.append({
                "pool_index": pool_idx,
                "source_qid": entry["source_qid"],
                "title": entry["title"],
            })
            return entry["text"]
        result = self.shuffle_pool[self.shuffle_idx % len(self.shuffle_pool)]
        self.shuffled_sources.append({"pool_index": self.shuffle_idx % len(self.shuffle_pool)})
        self.shuffle_idx += 1
        return result

    def _get_plausible_result(self, action: str, action_input: str) -> str:
        """Return a same-question distractor result when available."""
        if action == "search":
            if self.distractor_titles:
                return self._search_in_titles(
                    action_input, self.distractor_titles, force_result=True,
                )
            return self._get_shuffled_result()
        if action == "lookup":
            return self._lookup_real(action_input)
        return self._get_no_feedback_result(action)

    def _get_conflicting_result(
        self,
        action: str,
        action_input: str,
        real_result: str,
    ) -> str:
        """Inject one plausible distractor first, then restore real evidence."""
        if action == "search" and not self._conflict_seeded:
            self._conflict_seeded = True
            if self.distractor_titles:
                return self._search_in_titles(
                    action_input, self.distractor_titles, force_result=True,
                )
        return real_result

    def _get_no_feedback_result(self, action: str) -> str:
        """Return a generic no-information response."""
        if action == "search":
            return (
                "No relevant results found for your query. "
                "Please reason based on what you already know."
            )
        elif action == "lookup":
            return (
                "No matching content found. "
                "Please reason based on what you already know."
            )
        return "Action executed. No information available."

    def _finish(self, answer: str) -> str:
        self.finished = True
        self.final_answer = answer
        return f"Answer submitted: {answer}"

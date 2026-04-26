from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from .config import (
    RLConfig,
    VALID_CLAUSE_TYPES,
    VALID_RISK_LEVELS,
    VALID_SUGGESTED_ACTIONS,
)


@dataclass
class EpisodeState:
    previous_action_signature: Tuple[str, str, str] | None = None
    previous_clause_text: str = ""
    consecutive_same_action: int = 0
    suspicious_step_count: int = 0
    # Collapse detection: track how often each clause_type has been predicted.
    total_steps: int = 0
    clause_type_counts: Dict[str, int] = field(default_factory=dict)


# Common English words that carry no domain signal.
# Keeping this inline avoids any NLTK / spaCy dependency.
_STOPWORDS: frozenset = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "that", "this", "which", "who", "whom", "what", "where", "when",
    "and", "or", "but", "if", "not", "no", "nor", "so", "yet", "both",
    "either", "neither", "each", "any", "all", "few", "more", "most",
    "other", "some", "such", "than", "too", "very", "just", "it", "its",
    "their", "they", "them", "he", "she", "we", "us", "i", "my",
    "your", "our", "his", "her", "shall", "party", "parties", "agreement",
})


class RewardComposer:
    """Builds a composed reward from independent checks to reduce reward hacking risk."""

    def __init__(self, config: RLConfig):
        self.config = config

    @staticmethod
    def _extract_env_score(result: Dict[str, Any]) -> float:
        reward_obj = result.get("reward", {})
        if isinstance(reward_obj, dict):
            score = reward_obj.get("score", 0.0)
            return float(score) if score is not None else 0.0
        if isinstance(reward_obj, (int, float)):
            return float(reward_obj)
        return 0.0

    @staticmethod
    def _is_non_empty_string(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())

    @staticmethod
    def _content_words(text: str) -> frozenset:
        """Lowercase alphabetic tokens > 2 chars with stopwords removed."""
        tokens = re.findall(r"[a-z]+", text.lower())
        return frozenset(w for w in tokens if w not in _STOPWORDS and len(w) > 2)

    def _grounding_score(self, clause_text: str, reasoning: str) -> float:
        """
        Fraction of clause content-words that also appear in the reasoning.
        Returns 1.0 when the clause is empty (benefit of the doubt).
        """
        clause_words = self._content_words(clause_text)
        if not clause_words:
            return 1.0
        reasoning_words = self._content_words(reasoning)
        return len(clause_words & reasoning_words) / len(clause_words)

    def compose(
        self,
        action: Dict[str, Any],
        env_result: Dict[str, Any],
        state: EpisodeState,
        observation: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any], bool]:
        env_score = self._extract_env_score(env_result)

        clause_type = action.get("clause_type")
        risk_level = action.get("risk_level")
        suggested_action = action.get("suggested_action")
        reasoning = action.get("reasoning", "")

        schema_valid = int(
            action.get("action_type") == "classify"
            and self._is_non_empty_string(clause_type)
            and self._is_non_empty_string(risk_level)
            and self._is_non_empty_string(suggested_action)
            and isinstance(action.get("flags", []), list)
        )

        taxonomy_valid = int(
            clause_type in VALID_CLAUSE_TYPES
            and risk_level in VALID_RISK_LEVELS
            and suggested_action in VALID_SUGGESTED_ACTIONS
        )

        # Anti-gaming: only award the taxonomy bonus when the environment ALSO
        # agrees the classification is correct. A model that always outputs
        # "confidentiality / low / accept_as_is" will no longer earn free reward.
        taxonomy_bonus = (
            self.config.reward_taxonomy_bonus * taxonomy_valid
            if env_score >= self.config.taxonomy_bonus_min_env_score
            else 0.0
        )

        reasoning_len = len(reasoning.strip()) if isinstance(reasoning, str) else 0

        # Grounding check: does the reasoning actually reference words from the clause?
        # This blocks the model from earning process_valid by outputting nonsense
        # like "aaaaaaaaaaaaa" or a generic canned sentence.
        clause_text = str(observation.get("clause_text", ""))
        grounding_score = self._grounding_score(clause_text, reasoning)
        grounding_valid = int(grounding_score >= self.config.min_grounding_overlap)

        process_valid = int(
            self.config.min_reasoning_chars
            <= reasoning_len
            <= self.config.max_reasoning_chars
            and "no reasoning" not in reasoning.lower()
            and grounding_valid  # must reference actual clause content
        )

        action_signature = (
            str(clause_type),
            str(risk_level),
            str(suggested_action),
        )

        current_clause_text = str(observation.get("clause_text", ""))
        clause_changed = current_clause_text != state.previous_clause_text

        if action_signature == state.previous_action_signature:
            state.consecutive_same_action += 1
        else:
            state.consecutive_same_action = 0

        repeated_penalty = 0.0
        if state.consecutive_same_action >= self.config.repeated_action_soft_limit:
            repeated_penalty = self.config.reward_repeat_penalty

        drift_penalty = 0.0
        suspicious = False
        if (
            clause_changed
            and state.consecutive_same_action >= self.config.repeated_action_soft_limit
        ):
            drift_penalty = self.config.reward_drift_penalty
            suspicious = True
            state.suspicious_step_count += 1

        # Collapse penalty: if the model has learned to always predict the same
        # clause_type regardless of the clause content, penalise it.
        state.total_steps += 1
        state.clause_type_counts[str(clause_type)] = (
            state.clause_type_counts.get(str(clause_type), 0) + 1
        )
        collapse_penalty = 0.0
        if state.total_steps >= self.config.collapse_detection_min_steps:
            max_count = max(state.clause_type_counts.values())
            if (max_count / state.total_steps) > self.config.collapse_threshold:
                collapse_penalty = self.config.reward_collapse_penalty

        reward = (
            self.config.reward_env_weight * env_score
            + self.config.reward_schema_bonus * schema_valid
            + taxonomy_bonus
            + self.config.reward_process_bonus * process_valid
            - repeated_penalty
            - drift_penalty
            - collapse_penalty
        )
        reward = max(self.config.reward_min, min(self.config.reward_max, reward))

        force_stop = (
            state.consecutive_same_action >= self.config.repeated_action_hard_limit
        )

        state.previous_action_signature = action_signature
        state.previous_clause_text = current_clause_text

        columns = {
            "env_score": env_score,
            "schema_valid": schema_valid,
            "taxonomy_valid": taxonomy_valid,
            "taxonomy_bonus": taxonomy_bonus,
            "process_valid": process_valid,
            "grounding_score": round(grounding_score, 3),
            "grounding_valid": grounding_valid,
            "repeated_penalty": repeated_penalty,
            "drift_penalty": drift_penalty,
            "collapse_penalty": collapse_penalty,
            "composed_reward": reward,
            "suspicious": suspicious,
            "consecutive_same_action": state.consecutive_same_action,
        }
        return reward, columns, force_stop

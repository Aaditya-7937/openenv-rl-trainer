from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from .config import RLConfig, VALID_CLAUSE_TYPES, VALID_RISK_LEVELS, VALID_SUGGESTED_ACTIONS


@dataclass
class EpisodeState:
    previous_action_signature: Tuple[str, str, str] | None = None
    previous_clause_text: str = ""
    consecutive_same_action: int = 0
    suspicious_step_count: int = 0


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

        reasoning_len = len(reasoning.strip()) if isinstance(reasoning, str) else 0
        process_valid = int(
            self.config.min_reasoning_chars <= reasoning_len <= self.config.max_reasoning_chars
            and "no reasoning" not in reasoning.lower()
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
        if clause_changed and state.consecutive_same_action >= self.config.repeated_action_soft_limit:
            drift_penalty = self.config.reward_drift_penalty
            suspicious = True
            state.suspicious_step_count += 1

        reward = (
            self.config.reward_env_weight * env_score
            + self.config.reward_schema_bonus * schema_valid
            + self.config.reward_taxonomy_bonus * taxonomy_valid
            + self.config.reward_process_bonus * process_valid
            - repeated_penalty
            - drift_penalty
        )
        reward = max(self.config.reward_min, min(self.config.reward_max, reward))

        force_stop = state.consecutive_same_action >= self.config.repeated_action_hard_limit

        state.previous_action_signature = action_signature
        state.previous_clause_text = current_clause_text

        columns = {
            "env_score": env_score,
            "schema_valid": schema_valid,
            "taxonomy_valid": taxonomy_valid,
            "process_valid": process_valid,
            "repeated_penalty": repeated_penalty,
            "drift_penalty": drift_penalty,
            "composed_reward": reward,
            "suspicious": suspicious,
            "consecutive_same_action": state.consecutive_same_action,
        }
        return reward, columns, force_stop

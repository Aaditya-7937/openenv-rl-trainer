import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


VALID_CLAUSE_TYPES = [
    "indemnification",
    "limitation_of_liability",
    "termination",
    "confidentiality",
    "non_compete",
    "force_majeure",
    "assignment",
    "governing_law",
    "warranty",
    "intellectual_property",
    "payment_terms",
    "representations",
    "dispute_resolution",
    "data_protection",
    "insurance",
]
VALID_RISK_LEVELS = ["low", "medium", "high", "critical"]
VALID_SUGGESTED_ACTIONS = [
    "accept_as_is",
    "request_modification",
    "escalate_to_senior_counsel",
    "reject_clause",
    "flag_for_negotiation",
]


@dataclass
class RLConfig:
    # Environment Settings
    api_url: str = os.getenv(
        "SPACE_API_URL", "https://kamal1425-myspace.hf.space"
    )

    # Model Settings
    model_name: str = "google/gemma-4-26B-A4B-it"

    # Training Hyperparameters
    learning_rate: float = 1e-5
    grad_clip_norm: float = 1.0
    train_tasks: tuple = ("task_1_easy", "task_2_medium")
    eval_task: str = "task_3_hard"

    # Episode/Step limits
    max_steps_per_episode: int = 50
    total_training_episodes: int = 50

    # Reward shaping and verification (RLVR-style)
    reward_env_weight: float = float(os.getenv("REWARD_ENV_WEIGHT", "0.6"))
    reward_schema_bonus: float = float(os.getenv("REWARD_SCHEMA_BONUS", "0.15"))
    reward_taxonomy_bonus: float = float(os.getenv("REWARD_TAXONOMY_BONUS", "0.15"))
    reward_process_bonus: float = float(os.getenv("REWARD_PROCESS_BONUS", "0.1"))
    reward_repeat_penalty: float = float(os.getenv("REWARD_REPEAT_PENALTY", "0.12"))
    reward_drift_penalty: float = float(os.getenv("REWARD_DRIFT_PENALTY", "0.35"))
    reward_min: float = float(os.getenv("REWARD_MIN", "-5.0"))
    reward_max: float = float(os.getenv("REWARD_MAX", "5.0"))

    # Process checks and anti-hacking safety limits
    min_reasoning_chars: int = int(os.getenv("MIN_REASONING_CHARS", "20"))
    max_reasoning_chars: int = int(os.getenv("MAX_REASONING_CHARS", "600"))
    repeated_action_soft_limit: int = int(os.getenv("REPEATED_ACTION_SOFT_LIMIT", "2"))
    repeated_action_hard_limit: int = int(os.getenv("REPEATED_ACTION_HARD_LIMIT", "4"))

    # Monitoring and inspection
    inspect_every_n_steps: int = int(os.getenv("INSPECT_EVERY_N_STEPS", "5"))
    warn_if_suspicious_steps: int = int(os.getenv("WARN_IF_SUSPICIOUS_STEPS", "3"))

    # Reproducibility
    seed: int = int(os.getenv("SEED", "42"))

    # Generation settings
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "256"))
    train_do_sample: bool = _env_bool("TRAIN_DO_SAMPLE", True)
    eval_do_sample: bool = _env_bool("EVAL_DO_SAMPLE", False)
    train_temperature: float = float(os.getenv("TRAIN_TEMPERATURE", "0.7"))

    # Optional auth token for protected environment spaces
    env_api_key: str | None = os.getenv("OPENENV_API_KEY")

    @property
    def device(self) -> str:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

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
        "SPACE_API_URL", "https://aaditya-7937-openenv-review.hf.space"
    )

    # Model Settings (Using a highly capable 7B model that fits on single L4 24GB with LoRA)
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"

    # Training Hyperparameters
    learning_rate: float = 1e-5
    grad_clip_norm: float = 1.0
    train_tasks: tuple = ("task_1_easy", "task_2_medium")
    eval_task: str = "task_3_hard"

    # Episode/Step limits
    max_steps_per_episode: int = 10
    total_training_episodes: int = 2  # Keep this low for testing

    # Reproducibility
    seed: int = int(os.getenv("SEED", "42"))

    # Generation settings
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "128"))
    train_do_sample: bool = _env_bool("TRAIN_DO_SAMPLE", True)
    eval_do_sample: bool = _env_bool("EVAL_DO_SAMPLE", False)
    train_temperature: float = float(os.getenv("TRAIN_TEMPERATURE", "0.8"))
    top_p: float = float(os.getenv("TOP_P", "0.95"))

    # Optional auth token for protected environment spaces
    env_api_key: str | None = os.getenv("OPENENV_API_KEY")

    @property
    def device(self) -> str:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

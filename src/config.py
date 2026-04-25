import os
from dataclasses import dataclass

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
    train_tasks: tuple = ("task_1_easy", "task_2_medium")
    eval_task: str = "task_3_hard"

    # Episode/Step limits
    max_steps_per_episode: int = 10
    total_training_episodes: int = 2  # Keep this low for testing

    @property
    def device(self) -> str:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

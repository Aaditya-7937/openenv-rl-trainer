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
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"

    # Training Hyperparameters
    learning_rate: float = 5e-6  # Lower for 26B model stability with LoRA
    grad_clip_norm: float = 1.0
    # gamma (discount factor) removed — only needed by REINFORCE trajectory returns.
    # GRPO normalises within the group; no cross-step discounting required.

    train_tasks: tuple = ("task_1_easy", "task_2_medium")
    eval_task: str = "task_3_hard"

    # Episode/Step limits
    max_steps_per_episode: int = 10
    total_training_episodes: int = 2  # Keep this low for testing

    # GRPO (Group Relative Policy Optimization) settings
    grpo_group_size: int = int(os.getenv("GRPO_GROUP_SIZE", "4"))  # completions per prompt (generate_group)
    grpo_num_generations: int = int(os.getenv("GRPO_NUM_GENERATIONS", "4"))  # G for TRL GRPOTrainer / fallback loop
    grpo_samples_per_task: int = int(os.getenv("GRPO_SAMPLES_PER_TASK", "20"))  # dataset rows per task for TRL
    grpo_grad_accum_steps: int = int(os.getenv("GRPO_GRAD_ACCUM_STEPS", "2"))  # gradient accumulation
    grpo_kl_coeff: float = float(os.getenv("GRPO_KL_COEFF", "0.05"))  # KL penalty vs reference

    # Curriculum learning
    curriculum_unlock_threshold: float = float(os.getenv("CURRICULUM_UNLOCK_THRESHOLD", "0.3"))
    curriculum_window: int = int(os.getenv("CURRICULUM_WINDOW", "5"))

    # Safety
    episode_timeout_seconds: float = float(os.getenv("EPISODE_TIMEOUT_SECONDS", "300"))

    # Reward shaping and verification (RLVR-style)
    reward_env_weight: float = float(os.getenv("REWARD_ENV_WEIGHT", "0.6"))
    reward_schema_bonus: float = float(os.getenv("REWARD_SCHEMA_BONUS", "0.15"))
    reward_taxonomy_bonus: float = float(os.getenv("REWARD_TAXONOMY_BONUS", "0.15"))
    reward_process_bonus: float = float(os.getenv("REWARD_PROCESS_BONUS", "0.1"))
    reward_repeat_penalty: float = float(os.getenv("REWARD_REPEAT_PENALTY", "0.2"))
    reward_drift_penalty: float = float(os.getenv("REWARD_DRIFT_PENALTY", "0.15"))
    reward_min: float = float(os.getenv("REWARD_MIN", "-5.0"))
    reward_max: float = float(os.getenv("REWARD_MAX", "5.0"))

    # Process checks and anti-hacking safety limits
    min_reasoning_chars: int = int(os.getenv("MIN_REASONING_CHARS", "20"))
    max_reasoning_chars: int = int(os.getenv("MAX_REASONING_CHARS", "600"))
    # Grounding check: fraction of clause content-words that must appear in
    # the reasoning text. Prevents earning process_valid by outputting nonsense.
    min_grounding_overlap: float = float(os.getenv("MIN_GROUNDING_OVERLAP", "0.15"))
    repeated_action_soft_limit: int = int(os.getenv("REPEATED_ACTION_SOFT_LIMIT", "2"))
    repeated_action_hard_limit: int = int(os.getenv("REPEATED_ACTION_HARD_LIMIT", "4"))

    # Monitoring and inspection
    inspect_every_n_steps: int = int(os.getenv("INSPECT_EVERY_N_STEPS", "5"))
    warn_if_suspicious_steps: int = int(os.getenv("WARN_IF_SUSPICIOUS_STEPS", "3"))

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

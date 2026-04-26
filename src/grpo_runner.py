"""
GRPO training runner using TRL's GRPOTrainer.

GRPO (Group Relative Policy Optimization) is the preferred RL algorithm for
verifiable tasks.  For each prompt the trainer samples G completions, scores
all G via the reward function, then computes group-relative advantages:

    A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)

This eliminates the need for a separate value/critic network (unlike PPO) and
reduces variance compared to single-sample estimators, since the group mean
acts as a prompt-specific baseline.

Unsloth-loaded models are fully compatible with GRPOTrainer because Unsloth
patches the model in-place; the trainer just sees a standard HuggingFace model.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class GRPORunner:
    """
    Wraps TRL's GRPOTrainer for the OpenEnv contract review task.

    The dataset is built by sampling real observations from the environment
    so prompts always reflect actual clause text.  Each row holds:
      - "prompt"      : the full prompt string fed to the LLM
      - "task_id"     : the OpenEnv task ID (for env resets inside reward fn)
      - "clause_text" : raw clause text (for the grounding reward check)
    """

    def __init__(self, agent, config, client, reward_composer):
        # Eagerly validate that TRL + datasets are importable so failures
        # surface at construction time, not mid-training.
        from trl import GRPOConfig  # type: ignore  # noqa: F401
        from datasets import Dataset  # type: ignore  # noqa: F401

        self.agent = agent
        self.config = config
        self.client = client
        self.reward_composer = reward_composer

    # ------------------------------------------------------------------ #
    #  Dataset builder                                                     #
    # ------------------------------------------------------------------ #

    def build_dataset(self, task_ids: List[str], samples_per_task: int = 20):
        """
        Build a HuggingFace Dataset of prompt rows by sampling observations
        from the live environment.

        GRPOTrainer will sample `num_generations` completions for EACH row;
        the reward function is called once with all G completions per batch.
        """
        from datasets import Dataset  # type: ignore

        rows: List[Dict[str, Any]] = []
        for task_id in task_ids:
            collected = 0
            attempts = 0
            max_attempts = samples_per_task * 3  # tolerate env failures

            while collected < samples_per_task and attempts < max_attempts:
                attempts += 1
                try:
                    obs = self.client.reset(task_id)
                    clause_text = obs.get("clause_text", "")
                    if not clause_text:
                        continue  # skip empty observations
                    prompt = self.agent.create_prompt(obs)
                    rows.append(
                        {
                            "prompt": prompt,
                            "task_id": task_id,
                            "clause_text": clause_text,
                        }
                    )
                    collected += 1
                except Exception as exc:
                    logger.warning(f"[GRPORunner] Dataset sample failed ({task_id}): {exc}")

            print(
                f"[GRPORunner] Collected {collected}/{samples_per_task} "
                f"samples for task '{task_id}'"
            )

        if not rows:
            raise RuntimeError(
                "[GRPORunner] Dataset is empty — check environment connectivity."
            )

        return Dataset.from_list(rows)

    # ------------------------------------------------------------------ #
    #  Reward function factory                                             #
    # ------------------------------------------------------------------ #

    def make_reward_fn(self) -> Callable:
        """
        Returns a reward function with the signature expected by GRPOTrainer:

            fn(prompts, completions, **kwargs) -> List[float]

        For each completion the function:
          1. Parses the LLM text into an action dict.
          2. Resets the environment for that clause's task_id.
          3. Steps the environment with the parsed action.
          4. Composes the reward using RewardComposer (env + process checks).
          5. Returns 0.0 on parse failure or environment error.

        Note: Each completion gets its OWN env.reset() call so that G parallel
        rollouts are independent.  This is necessary because the OpenEnv
        environment is stateful and does not support forked sessions.
        """
        from .rewarding import EpisodeState  # local import to avoid circular

        agent = self.agent
        client = self.client
        reward_composer = self.reward_composer
        default_task = self.config.train_tasks[0]

        def reward_fn(
            prompts: List[str],
            completions: List[str],
            **kwargs,
        ) -> List[float]:
            clause_texts: List[str] = kwargs.get(
                "clause_text", [""] * len(completions)
            )
            task_ids: List[str] = kwargs.get(
                "task_id", [default_task] * len(completions)
            )

            rewards: List[float] = []
            for completion, clause_text, task_id in zip(
                completions, clause_texts, task_ids
            ):
                try:
                    # Parse LLM output → action dict
                    action = agent.parse_action(completion)
                    parse_failed = action.pop("_parse_failed", False)
                    action.pop("_raw_generation", "")  # strip internal tags

                    if parse_failed:
                        rewards.append(0.0)
                        continue

                    # Independent env rollout for this completion
                    obs = client.reset(task_id)
                    result = client.step(action)

                    # Fresh EpisodeState so rollouts don't share anti-hack counters
                    state = EpisodeState()
                    observation = {"clause_text": clause_text}

                    reward, _, _ = reward_composer.compose(
                        action=action,
                        env_result=result,
                        state=state,
                        observation=observation,
                    )
                    rewards.append(float(reward))

                except Exception as exc:
                    logger.warning(f"[GRPORunner] Reward fn error: {exc}")
                    rewards.append(0.0)

            # Log group reward statistics for monitoring
            if rewards:
                mean_r = sum(rewards) / len(rewards)
                max_r = max(rewards)
                min_r = min(rewards)
                print(
                    f"[GRPORunner] Batch rewards — "
                    f"mean={mean_r:.3f} max={max_r:.3f} min={min_r:.3f}"
                )
            return rewards

        return reward_fn

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def run(self, task_ids: List[str], samples_per_task: int = 20) -> None:
        """
        Build the prompt dataset, instantiate GRPOTrainer, and run training.

        Args:
            task_ids:         List of OpenEnv task IDs to sample from.
            samples_per_task: Number of prompt rows to collect per task.
                              GRPOTrainer generates `num_generations` completions
                              per row, so total rollouts = samples_per_task
                              × len(task_ids) × num_generations.
        """
        from trl import GRPOTrainer, GRPOConfig  # type: ignore

        print(
            f"[GRPORunner] Building dataset: tasks={task_ids}, "
            f"{samples_per_task} samples/task"
        )
        dataset = self.build_dataset(task_ids, samples_per_task)

        total_rollouts = len(dataset) * self.config.grpo_num_generations
        print(
            f"[GRPORunner] Dataset: {len(dataset)} prompts × "
            f"{self.config.grpo_num_generations} generations = "
            f"{total_rollouts} total rollouts"
        )

        # NOTE: TRL ≥0.9 / Unsloth renamed `max_new_tokens` → `max_completion_length`.
        # per_device_train_batch_size must be a multiple of num_generations (Unsloth rule).
        grpo_config = GRPOConfig(
            output_dir="./results/grpo_checkpoints",
            num_generations=self.config.grpo_num_generations,
            max_completion_length=self.config.max_new_tokens,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.grpo_num_generations,  # must be multiple of num_generations
            gradient_accumulation_steps=self.config.grpo_grad_accum_steps,
            num_train_epochs=1,
            max_grad_norm=self.config.grad_clip_norm,
            seed=self.config.seed,
            logging_steps=1,
            save_steps=50,
            remove_unused_columns=False,
            temperature=self.config.train_temperature,
            top_p=self.config.top_p,
        )

        reward_fn = self.make_reward_fn()

        trainer = GRPOTrainer(
            model=self.agent.model,
            reward_funcs=reward_fn,
            args=grpo_config,
            train_dataset=dataset,
            processing_class=self.agent.tokenizer,
        )

        print("[GRPORunner] Starting GRPOTrainer.train() ...")
        trainer.train()
        print("[GRPORunner] Training complete.")

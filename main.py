import os
import random
import time
import torch
from dotenv import load_dotenv
from src.config import RLConfig
from src.env_client import EnvironmentClient
from src.agent import RLAgent
from src.evaluation import Evaluator
from src.rewarding import RewardComposer, EpisodeState


def main():
    # 1. Load configuration and initialize components.
    # Use the .env file explicitly located in the current folder (openenv_rl_trainer)
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    config = RLConfig()

    # Reproducibility for easier debugging and fair pre/post comparisons.
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    client = EnvironmentClient(api_url=config.api_url, api_key=config.env_api_key)
    agent = RLAgent(config=config)
    evaluator = Evaluator()
    reward_composer = RewardComposer(config=config)

    print("=" * 60)
    print("🚀 OPENENV REINFORCEMENT LEARNING PIPELINE STARTING 🚀")
    print(f"Model: {config.model_name}")
    print(f"Device: {agent.device}")
    print(f"Seed: {config.seed}")
    print(
        "Reward Weights: "
        f"env={config.reward_env_weight}, "
        f"schema={config.reward_schema_bonus}, "
        f"taxonomy={config.reward_taxonomy_bonus}, "
        f"process={config.reward_process_bonus}"
    )
    print("=" * 60)

    # 2. Pre-Training Evaluation: Test untrainned model on 'hard' task
    print("\nPhase 1: Baseline Evaluation (Testing the out-of-the-box model)")
    baseline_score = evaluator.run_evaluation(agent, client, task_id=config.eval_task)
    evaluator.metrics["pre_eval"] = baseline_score
    print(f"Baseline Score: {baseline_score}")

    # 3. Training Loop
    print("\nPhase 2: Reinforcement Learning")

    # ── Primary path: TRL GRPOTrainer ────────────────────────────────────────
    # GRPO samples G completions per prompt, normalises rewards within the group,
    # and updates weights once — no value network needed, lower variance than
    # vanilla REINFORCE.  Requires: trl>=0.9.0 + datasets.
    _grpo_used = False
    try:
        from src.grpo_runner import GRPORunner

        print("[GRPO] TRL detected ✓ — using GRPOTrainer (preferred stack).")
        grpo = GRPORunner(
            agent=agent,
            config=config,
            client=client,
            reward_composer=reward_composer,
        )

        # Curriculum Phase A: easy tasks only
        print(
            f"\n[Curriculum] Phase 2a: GRPO on easy tasks "
            f"({config.grpo_samples_per_task} samples × "
            f"{config.grpo_num_generations} generations)..."
        )
        grpo.run(
            task_ids=[config.train_tasks[0]],
            samples_per_task=config.grpo_samples_per_task,
        )

        # Curriculum gate: evaluate on easy task; unlock medium if threshold met
        easy_eval = evaluator.run_evaluation(
            agent, client, task_id=config.train_tasks[0]
        )
        evaluator.metrics["curriculum_easy_eval"] = easy_eval
        print(
            f"[Curriculum] Easy task eval score: {easy_eval:.3f} "
            f"(threshold: {config.curriculum_unlock_threshold})"
        )

        if len(config.train_tasks) > 1:
            if easy_eval >= config.curriculum_unlock_threshold:
                print(
                    f"[Curriculum] \U0001f393 Threshold met — unlocking medium tasks!"
                )
                grpo.run(
                    task_ids=list(config.train_tasks),
                    samples_per_task=config.grpo_samples_per_task,
                )
            else:
                print(
                    "[Curriculum] Threshold not met — keeping medium tasks locked. "
                    "Running a second easy-task GRPO pass instead."
                )
                grpo.run(
                    task_ids=[config.train_tasks[0]],
                    samples_per_task=config.grpo_samples_per_task,
                )

        _grpo_used = True

    except ImportError as exc:
        print(
            f"[Warning] GRPO path unavailable ({exc}). "
            "Install trl>=0.9.0 and datasets for the full stack. "
            "Falling back to PyTorch-native GRPO loop."
        )

    # ── Fallback path: PyTorch-native GRPO ──────────────────────────────────
    if not _grpo_used:
        # --- Curriculum state ---
        # Track rolling average reward on the easy task; unlock harder tasks only
        # once the model proves it can get consistent non-zero reward on easy ones.
        easy_reward_window: list = []
        medium_unlocked: bool = False

        for episode in range(config.total_training_episodes):
            for task_id in config.train_tasks:
                # --- Curriculum gate ---
                is_easy = task_id == config.train_tasks[0]
                if not is_easy and not medium_unlocked:
                    print(
                        f"[Curriculum] Skipping '{task_id}' — "
                        f"locked until easy avg reward ≥ {config.curriculum_unlock_threshold}"
                    )
                    continue

                # --- Checkpoint: save weights BEFORE the episode for rollback ---
                checkpoint_path = os.path.join(
                    "results", "checkpoints", f"last_good_{task_id}.pt"
                )
                agent.save_checkpoint(checkpoint_path)

                episode_rewards: list = []  # mean reward per step, for curriculum tracking
                episode_start_time = time.time()

                # GRPO: for each step, generate G completions from the SAME observation,
                # score all G independently, compute group-relative advantages, update once.
                # Each rollout gets its own env.reset() because OpenEnv is stateful.
                for step_count in range(config.max_steps_per_episode):

                    # Wall-clock timeout guard
                    if time.time() - episode_start_time > config.episode_timeout_seconds:
                        print(
                            f"[Safety] Episode timeout "
                            f"({time.time()-episode_start_time:.0f}s). Stopping."
                        )
                        break

                    # Get a fresh observation for this group rollout
                    try:
                        obs = client.reset(task_id)
                    except Exception as exc:
                        print(f"[GRPO] Env reset failed at step {step_count+1}: {exc}")
                        break

                    clause = obs.get("clause_text", "")
                    if not clause:
                        break

                    prompt = agent.create_prompt(obs)

                    # ── Generate G completions for this prompt ─────────────
                    group_log_probs = []
                    group_rewards = []
                    last_columns = {}
                    last_raw = ""

                    for g in range(config.grpo_num_generations):
                        if time.time() - episode_start_time > config.episode_timeout_seconds:
                            break

                        action, log_prob = agent.generate_and_get_logprobs(prompt)
                        parse_failed = action.pop("_parse_failed", False)
                        raw_gen = action.pop("_raw_generation", "")
                        if g == 0:
                            last_raw = raw_gen  # log first generation for inspection

                        if parse_failed:
                            group_log_probs.append(log_prob)
                            group_rewards.append(0.0)
                            continue

                        # Independent env interaction for rollout g
                        try:
                            client.reset(task_id)       # reset to same task state
                            result = client.step(action)
                        except Exception as exc:
                            print(f"[GRPO] Rollout {g} env error: {exc}")
                            group_log_probs.append(log_prob)
                            group_rewards.append(0.0)
                            continue

                        fresh_state = EpisodeState()
                        reward, columns, _ = reward_composer.compose(
                            action=action,
                            env_result=result,
                            state=fresh_state,
                            observation={"clause_text": clause},
                        )
                        group_log_probs.append(log_prob)
                        group_rewards.append(reward)
                        last_columns = columns

                    if not group_log_probs:
                        continue

                    # ── GRPO update ────────────────────────────────────────
                    # A_i = (r_i - mean(r)) / std(r)  — no critic needed
                    advantages = agent.compute_grpo_advantages(group_rewards)
                    agent.update_model_grpo(group_log_probs, advantages)

                    mean_r = sum(group_rewards) / len(group_rewards)
                    best_r = max(group_rewards)
                    episode_rewards.append(mean_r)
                    evaluator.record_training_reward(mean_r)
                    if last_columns:
                        evaluator.record_training_step(
                            {
                                "episode": episode + 1,
                                "task_id": task_id,
                                "step": step_count + 1,
                                **last_columns,
                            }
                        )

                    print(
                        f"[GRPO] Ep {episode+1} | {task_id} | "
                        f"Step {step_count+1}/{config.max_steps_per_episode} | "
                        f"G={len(group_log_probs)} | "
                        f"Mean={mean_r:.3f} Best={best_r:.3f}"
                    )

                    # Inspect block: show reward breakdown + raw generation
                    if (step_count + 1) % max(config.inspect_every_n_steps, 1) == 0 and last_columns:
                        print(
                            "[Inspect] "
                            f"schema={last_columns.get('schema_valid')} "
                            f"taxonomy={last_columns.get('taxonomy_valid')} "
                            f"(bonus={last_columns.get('taxonomy_bonus', 0):.2f}) "
                            f"process={last_columns.get('process_valid')} "
                            f"grounding={last_columns.get('grounding_score', 0):.2f} "
                            f"collapse_penalty={last_columns.get('collapse_penalty', 0):.2f}"
                        )
                        print(f"[Inspect] Raw generation[0]: {repr(last_raw[:300])}")

                # ── Post-episode: rollback + curriculum update ─────────────
                if episode_rewards:
                    episode_mean = sum(episode_rewards) / len(episode_rewards)

                    # Rollback if episode was clearly regressing
                    if episode_mean < -0.1:
                        print(
                            f"[Safety] Episode mean reward {episode_mean:.3f} < -0.1. "
                            "Rolling back to pre-episode checkpoint."
                        )
                        agent.load_checkpoint(checkpoint_path)

                    # Curriculum update (easy task only)
                    if is_easy:
                        easy_reward_window.append(episode_mean)
                        recent = easy_reward_window[-config.curriculum_window :]
                        recent_avg = sum(recent) / len(recent)
                        print(
                            f"[Curriculum] Easy task mean reward: {episode_mean:.3f} "
                            f"| Rolling avg (window={len(recent)}): {recent_avg:.3f} "
                            f"| Threshold: {config.curriculum_unlock_threshold}"
                        )
                        if not medium_unlocked and recent_avg >= config.curriculum_unlock_threshold:
                            medium_unlocked = True
                            print(
                                f"[Curriculum] \U0001f393 Medium tasks UNLOCKED at episode {episode + 1}!"
                            )

    # 4. Post-Training Evaluation: Test trained model on 'hard' task again
    print("\nPhase 3: Post-Training Blind Evaluation")
    trained_score = evaluator.run_evaluation(agent, client, task_id=config.eval_task)
    evaluator.metrics["post_eval"] = trained_score
    print(f"Trained Score: {trained_score}")

    # 5. Output Graphics & Results
    print("\nPhase 4: Generating Metrics & Visuals")
    evaluator.plot_and_save(save_dir="./results")

    print(
        "\n✅ Training Pipeline Complete! Please check the './results' folder for your graphs."
    )


if __name__ == "__main__":
    main()

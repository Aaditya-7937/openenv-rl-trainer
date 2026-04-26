import os
import random
import torch
from dotenv import load_dotenv
from src.config import RLConfig
from src.env_client import EnvironmentClient
from src.agent import RLAgent
from src.evaluation import Evaluator
from src.rewarding import RewardComposer, EpisodeState


def process_task(task_id, episode, config, agent, reward_composer, evaluator):
    """
    Process a single task episode with proper clause navigation:
      classify → next_clause → classify → ... → complete_review
    """
    client = EnvironmentClient(api_url=config.api_url, api_key=config.env_api_key)
    obs = client.reset(task_id)

    total_clauses = obs.get("total_clauses", 1)
    step_count = 0
    state = EpisodeState()
    episode_rewards = []

    for clause_idx in range(total_clauses):
        if step_count >= config.max_steps_per_episode:
            print(f"  [Limit] Max steps ({config.max_steps_per_episode}) reached.")
            break

        clause = obs.get("clause_text", "")
        if not clause:
            # No clause text means we should complete the review
            break

        # --- 1. LLM generates a classification for the current clause ---
        prompt = agent.create_prompt(obs)
        current_obs = dict(obs)
        action, log_probs, generated_text = agent.generate_and_get_logprobs(prompt)

        # --- 2. Send the classify action to the environment ---
        classify_result = client.step(action)
        classify_obs = classify_result.get("observation", {})
        classify_done = classify_result.get("done", False)

        # --- 3. Compose the reward from independent checks ---
        reward, columns, force_stop = reward_composer.compose(
            action=action,
            env_result=classify_result,
            state=state,
            observation=current_obs,
        )
        episode_rewards.append(reward)

        # --- 4. Print step details ---
        # Show raw generation so we can diagnose parse failures / mode collapse
        gen_preview = generated_text[:200].replace('\n', ' ').strip()
        print(
            f"  Clause {clause_idx+1}/{total_clauses} | "
            f"type={action.get('clause_type')} | "
            f"risk={action.get('risk_level')} | "
            f"action={action.get('suggested_action')} | "
            f"reward={reward:.3f} (env={columns['env_score']:.3f})"
        )
        print(f"    Raw: {gen_preview}")
        # Show env corrective feedback if any
        env_feedback = classify_obs.get("corrective_feedback", "")
        if env_feedback:
            print(f"    Feedback: {env_feedback[:150]}")

        # --- 5. Update model weights (REINFORCE) ---
        agent.update_model(log_probs, reward)
        evaluator.record_training_reward(reward)
        evaluator.record_training_step(
            {
                "episode": episode + 1,
                "task_id": task_id,
                "clause_index": clause_idx,
                "step": step_count + 1,
                **columns,
            }
        )

        step_count += 1

        if force_stop:
            print("  [Safety] Stopping episode early due to repeated identical actions.")
            break

        if classify_done:
            break

        # --- 6. Navigate to next clause or complete the review ---
        if clause_idx < total_clauses - 1:
            # Advance to the next clause
            next_result = client.step({"action_type": "next_clause"})
            obs = next_result.get("observation", {})
            if next_result.get("done", False):
                break
        else:
            # All clauses classified — complete the review
            complete_result = client.step({"action_type": "complete_review"})
            final_reward_obj = complete_result.get("reward", {})
            final_score = (
                final_reward_obj.get("score", 0.0)
                if isinstance(final_reward_obj, dict)
                else float(final_reward_obj or 0.0)
            )
            print(f"  [Complete] Final grader score for {task_id}: {final_score:.4f}")

    avg_reward = sum(episode_rewards) / max(len(episode_rewards), 1)
    print(f"  Episode avg reward: {avg_reward:.3f} over {len(episode_rewards)} steps")


def main():
    # 1. Load configuration and initialize components.
    # Try loading .env from this directory first, then fallback to openev directory
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    # Also load from the openev workspace if available (for shared secrets)
    openev_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "openev", ".env")
    if os.path.exists(openev_env):
        load_dotenv(openev_env, override=False)
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
    print(f"Max new tokens: {config.max_new_tokens}")
    print(f"Training episodes: {config.total_training_episodes}")
    print(f"Train tasks: {config.train_tasks}")
    print(f"Eval task: {config.eval_task}")
    print(
        "Reward Weights: "
        f"env={config.reward_env_weight}, "
        f"schema={config.reward_schema_bonus}, "
        f"taxonomy={config.reward_taxonomy_bonus}, "
        f"process={config.reward_process_bonus}"
    )
    print("=" * 60)

    # 2. Pre-Training Evaluation: Test untrained model on 'hard' task
    print("\nPhase 1: Baseline Evaluation (Testing the out-of-the-box model)")
    baseline_score = evaluator.run_evaluation(agent, client, task_id=config.eval_task)
    evaluator.metrics["pre_eval"] = baseline_score
    print(f"Baseline Score: {baseline_score}")

    # 3. Training Loop: Sequential processing with proper clause navigation
    print("\nPhase 2: Reinforcement Learning (REINFORCE)")
    for episode in range(config.total_training_episodes):
        print(f"\n{'─'*50}")
        print(f"Episode {episode+1}/{config.total_training_episodes}")
        print(f"{'─'*50}")

        for task_id in config.train_tasks:
            print(f"\n[Train] Task: {task_id}")
            try:
                process_task(
                    task_id, episode, config, agent, reward_composer, evaluator
                )
            except Exception as e:
                print(f"  [Error] Task {task_id} failed: {e}")

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

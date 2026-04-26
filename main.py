import os
import random
import torch
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from src.config import RLConfig
from src.env_client import EnvironmentClient
from src.agent import RLAgent
from src.evaluation import Evaluator
from src.rewarding import RewardComposer, EpisodeState


def process_task(task_id, episode, config, agent, reward_composer, evaluator, agent_lock):
    # Use a separate client for each parallel task to avoid session state clobbering
    client = EnvironmentClient(api_url=config.api_url, api_key=config.env_api_key)
    obs = client.reset(task_id)
    done = False
    step_count = 0
    state = EpisodeState()

    while not done and step_count < config.max_steps_per_episode:
        clause = obs.get("clause_text", "")
        if not clause:
            client.step({"action_type": "complete_review"})
            break

        # 3a. LLM analyzes the text
        prompt = agent.create_prompt(obs)
        current_obs = dict(obs)

        # 3b. LLM generates action dict AND we calculate token probabilties for gradient math
        with agent_lock:
            action, log_probs, generated_text = agent.generate_and_get_logprobs(prompt)

        # 3c. Send action to OpenEnv and receive reward (This happens in parallel!)
        result = client.step(action)
        obs = result.get("observation", {})
        done = result.get("done", False)

        # 3d. Compose reward from independent checks (outcome + process + safety)
        reward, columns, force_stop = reward_composer.compose(
            action=action,
            env_result=result,
            state=state,
            observation=current_obs,
        )

        with agent_lock:
            print(
                f"\n[Train] Episode {episode+1} | Task: {task_id} | Step {step_count+1}"
            )
            print(f"--- Prompt Sample ---\n{prompt[:150]}...\n---------------------")
            print(f"--- Output Sample ---\n{generated_text}\n---------------------")
            print(
                f"| Reward: {reward:.3f} | Env: {columns['env_score']:.3f} "
                f"| Pred: {action.get('clause_type')}"
            )

            # 3e. MATHEMATICAL WEIGHT UPDATE (The actual learning happens here!)
            agent.update_model(log_probs, reward)
            evaluator.record_training_reward(reward)
            evaluator.record_training_step(
                {
                    "episode": episode + 1,
                    "task_id": task_id,
                    "step": step_count + 1,
                    **columns,
                }
            )

        if (step_count + 1) % max(config.inspect_every_n_steps, 1) == 0:
            with agent_lock:
                print(
                    "[Inspect] "
                    f"schema={columns['schema_valid']} "
                    f"taxonomy={columns['taxonomy_valid']} "
                    f"process={columns['process_valid']} "
                    f"suspicious_steps={state.suspicious_step_count}"
                )

        if state.suspicious_step_count >= config.warn_if_suspicious_steps:
            with agent_lock:
                print(
                    "[Warning] Repeated suspicious behavior detected. "
                    "Consider reducing temperature or tightening verifier checks."
                )

        if force_stop:
            with agent_lock:
                print(
                    "[Safety] Stopping episode early due to repeated identical actions."
                )
            done = True

        step_count += 1


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

    # 3. Training Loop: Let the model practice on easy/medium tasks and update weights mathematically
    print("\nPhase 2: Reinforcement Learning (PPO/REINFORCE mechanism) with Parallel Processing")
    agent_lock = threading.Lock()

    for episode in range(config.total_training_episodes):
        with ThreadPoolExecutor(max_workers=max(4, len(config.train_tasks))) as executor:
            futures = [
                executor.submit(process_task, task_id, episode, config, agent, reward_composer, evaluator, agent_lock)
                for task_id in config.train_tasks
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Task failed: {e}")

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

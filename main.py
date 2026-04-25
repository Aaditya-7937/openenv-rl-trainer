import os
import random
import torch
from dotenv import load_dotenv
from src.config import RLConfig
from src.env_client import EnvironmentClient
from src.agent import RLAgent
from src.evaluation import Evaluator


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

    print("=" * 60)
    print("🚀 OPENENV REINFORCEMENT LEARNING PIPELINE STARTING 🚀")
    print(f"Model: {config.model_name}")
    print(f"Device: {agent.device}")
    print(f"Seed: {config.seed}")
    print("=" * 60)

    # 2. Pre-Training Evaluation: Test untrainned model on 'hard' task
    print("\nPhase 1: Baseline Evaluation (Testing the out-of-the-box model)")
    baseline_score = evaluator.run_evaluation(agent, client, task_id=config.eval_task)
    evaluator.metrics["pre_eval"] = baseline_score
    print(f"Baseline Score: {baseline_score}")

    # 3. Training Loop: Let the model practice on easy/medium tasks and update weights mathematically
    print("\nPhase 2: Reinforcement Learning (PPO/REINFORCE mechanism)")
    for episode in range(config.total_training_episodes):
        for task_id in config.train_tasks:
            obs = client.reset(task_id)
            done = False
            step_count = 0

            while not done and step_count < config.max_steps_per_episode:
                clause = obs.get("clause_text", "")
                if not clause:
                    resp = client.step({"action_type": "complete_review"})
                    break

                # 3a. LLM analyzes the text
                prompt = agent.create_prompt(obs)

                # 3b. LLM generates action dict AND we calculate token probabilties for gradient math
                action, log_probs = agent.generate_and_get_logprobs(prompt)

                # 3c. Send action to OpenEnv and receive reward
                result = client.step(action)
                obs = result.get("observation", {})
                score = result.get("reward", {}).get("score", 0.0)
                done = result.get("done", False)

                print(
                    f"[Train] Episode {episode+1} | Task: {task_id} | Step {step_count+1} | Reward: {score} | Pred: {action.get('clause_type')}"
                )

                # 3d. MATHEMATICAL WEIGHT UPDATE (The actual learning happens here!)
                if score is not None:
                    agent.update_model(log_probs, score)
                    evaluator.record_training_reward(score)

                step_count += 1

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

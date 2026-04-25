import matplotlib.pyplot as plt
import json
import os
from typing import List, Dict


class Evaluator:
    """Handles Blind Evaluation and Metrics Plotting."""

    def __init__(self):
        self.metrics = {"pre_eval": 0.0, "post_eval": 0.0, "training_rewards": []}

    def record_training_reward(self, reward: float):
        self.metrics["training_rewards"].append(reward)

    def run_evaluation(self, agent, env_client, task_id: str) -> float:
        """Runs the model on the environment without updating weights."""
        print(f"\n--- Running Blind Evaluation on {task_id} ---")

        # Tell PyTorch NOT to track gradients (This prevents learning/overfitting)
        import torch

        with torch.no_grad():
            obs = env_client.reset(task_id)
            done = False
            total_reward = 0.0

            while not done:
                clause = obs.get("clause_text", "")
                if not clause:
                    resp = env_client.step({"action_type": "complete_review"})
                    done = resp.get("done", True)
                    score = resp.get("reward", {}).get("score", 0.0)
                    total_reward += score
                    break

                prompt = agent.create_prompt(obs)
                inputs = agent.tokenizer(prompt, return_tensors="pt").to(agent.device)

                output = agent.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=128,
                    do_sample=True,
                    pad_token_id=agent.tokenizer.pad_token_id,
                )

                generated_text = agent.tokenizer.decode(
                    output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                )
                action = agent.parse_action(generated_text)

                result = env_client.step(action)
                obs = result.get("observation", {})
                score = result.get("reward", {}).get("score", 0.0)
                done = result.get("done", False)

                total_reward += score
                print(f"Eval Clause Score: {score}")

        return total_reward

    def plot_and_save(self, save_dir: str = "."):
        """Generate visualizations for the training results."""
        os.makedirs(save_dir, exist_ok=True)

        # 1. Pre vs Post Eval Bar Chart
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        bars = plt.bar(
            ["Untrained Model", "Trained Model"],
            [self.metrics["pre_eval"], self.metrics["post_eval"]],
            color=["red", "green"],
        )
        plt.title("Evaluation on Test Set (Task 3)")
        plt.ylabel("Total Reward Score")

        # 2. Training Rewards Line Chart
        plt.subplot(1, 2, 2)
        plt.plot(
            self.metrics["training_rewards"], marker="o", color="blue", linestyle="-"
        )
        plt.title("Reward Trajectory Over Training Steps")
        plt.xlabel("Step")
        plt.ylabel("Reward")

        plt.tight_layout()
        viz_path = os.path.join(save_dir, "training_results.png")
        plt.savefig(viz_path)
        print(f"\n[Evaluator] Generated graphs and saved to {viz_path}")

        # Save exact json metrics
        with open(os.path.join(save_dir, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=4)

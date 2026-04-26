import matplotlib.pyplot as plt
import json
import os
from typing import Dict, Any


class Evaluator:
    """Handles Blind Evaluation and Metrics Plotting."""

    def __init__(self):
        self.metrics = {
            "pre_eval": 0.0,
            "post_eval": 0.0,
            "training_rewards": [],
            "training_steps": [],
            "reward_columns_mean": {},
        }

    def record_training_reward(self, reward: float):
        self.metrics["training_rewards"].append(reward)

    def record_training_step(self, row: Dict[str, Any]):
        self.metrics["training_steps"].append(row)

    def finalize_training_summary(self):
        steps = self.metrics.get("training_steps", [])
        if not steps:
            self.metrics["reward_columns_mean"] = {}
            return

        numeric_cols = [
            "env_score",
            "schema_valid",
            "taxonomy_valid",
            "process_valid",
            "repeated_penalty",
            "drift_penalty",
            "composed_reward",
        ]
        summary = {}
        for key in numeric_cols:
            values = [float(step.get(key, 0.0)) for step in steps]
            summary[key] = sum(values) / max(len(values), 1)
        summary["suspicious_rate"] = sum(
            int(bool(step.get("suspicious", False))) for step in steps
        ) / max(len(steps), 1)
        self.metrics["reward_columns_mean"] = summary

    def run_evaluation(self, agent, env_client, task_id: str) -> float:
        """Runs the model on the environment without updating weights."""
        print(f"\n--- Running Blind Evaluation on {task_id} ---")

        # Tell PyTorch NOT to track gradients (This prevents learning/overfitting)
        import torch

        was_training = agent.model.training
        agent.model.eval()
        try:
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
                    inputs = agent.tokenizer(prompt, return_tensors="pt").to(
                        agent.device
                    )

                    generation_kwargs = {
                        "max_new_tokens": agent.config.max_new_tokens,
                        "do_sample": agent.config.eval_do_sample,
                        "pad_token_id": agent.tokenizer.pad_token_id,
                    }
                    if agent.config.eval_do_sample:
                        generation_kwargs["temperature"] = (
                            agent.config.train_temperature
                        )
                        generation_kwargs["top_p"] = agent.config.top_p

                    output = agent.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        **generation_kwargs,
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
        finally:
            if was_training:
                agent.model.train()

        return total_reward

    def plot_and_save(self, save_dir: str = "."):
        """Generate visualizations for the training results."""
        os.makedirs(save_dir, exist_ok=True)
        self.finalize_training_summary()

        # 1. Pre vs Post Eval Bar Chart
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 3, 1)
        bars = plt.bar(
            ["Untrained Model", "Trained Model"],
            [self.metrics["pre_eval"], self.metrics["post_eval"]],
            color=["red", "green"],
        )
        plt.title("Evaluation on Test Set (Task 3)")
        plt.ylabel("Total Reward Score")

        # 2. Training Rewards Line Chart
        plt.subplot(1, 3, 2)
        plt.plot(
            self.metrics["training_rewards"], marker="o", color="blue", linestyle="-"
        )
        plt.title("Reward Trajectory Over Training Steps")
        plt.xlabel("Step")
        plt.ylabel("Reward")

        # 3. Component means for reward debugging
        summary = self.metrics.get("reward_columns_mean", {})
        keys = ["env_score", "schema_valid", "taxonomy_valid", "process_valid"]
        vals = [summary.get(k, 0.0) for k in keys]
        plt.subplot(1, 3, 3)
        plt.bar(keys, vals, color=["#1d4ed8", "#0f766e", "#166534", "#a16207"])
        plt.xticks(rotation=30, ha="right")
        plt.title("Mean Reward Columns")
        plt.ylabel("Mean Value")

        plt.tight_layout()
        viz_path = os.path.join(save_dir, "training_results.png")
        plt.savefig(viz_path)
        print(f"\n[Evaluator] Generated graphs and saved to {viz_path}")

        # Save exact json metrics
        with open(os.path.join(save_dir, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=4)

---
title: OpenEnv RL Trainer
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: gradio
python_version: "3.10"
app_file: app.py
suggested_hardware: a100-large
pinned: false
---

# OpenEnv RL Trainer

This folder contains a fully modular, PyTorch-first Reinforcement Learning (RL) training pipeline for the OpenEnv Contract Review benchmark, built on the **TRL + Unsloth + OpenEnv** hackathon stack.

## Technology Stack

| Component | Role |
|---|---|
| **OpenEnv** | Standardised environment interface (`/reset`, `/step`, reward signals) |
| **TRL** (`trl>=0.9.0`) | RL trainers — `GRPOTrainer` with group-relative advantage (GRPO) |
| **Unsloth** | 2× faster training + ~60% VRAM reduction (NF4 quant + fused Triton kernels) |
| **PEFT / LoRA** | Automatic fallback if Unsloth is unavailable |

It is now set up as an RLVR-style pipeline (Reinforcement Learning with Verifiable Rewards):
- reward is not only one scalar from the environment,
- a local verifier adds independent checks (schema, taxonomy, process quality),
- anti-hacking penalties discourage degenerate repeated outputs.

## Architecture

The project is structured using clean Software Design Principles (Separation of Concerns, Dependency Injection, Modularity).

### Files:
- \`main.py\`: The plug-and-play entrypoint orchestrating the experiment.
- \`app.py\`: Lightweight FastAPI service to trigger training and fetch logs/results.
- \`src/config.py\`: Centralized hyperparameters and settings.
- \`src/env_client.py\`: Handles HTTP communication with the Hugging Face space.
- \`src/agent.py\`: The AI Agent containing the LLM, tokenization, and RL optimization logic (Policy Gradient).
- \`src/evaluation.py\`: Logic for running blind evaluations, collecting metrics, and plotting graphs.
- \`src/rewarding.py\`: Verifier-style composed reward logic and anti-hacking safeguards.

## How to Run

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
   > **Unsloth note**: Unsloth requires a matching PyTorch + CUDA wheel.
   > If the default install fails, use the environment-specific wheel:
   > ```bash
   > # Example for CUDA 12.1 + PyTorch 2.3
   > pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
   > ```
   > The agent automatically falls back to standard HF + PEFT LoRA if Unsloth is not available.

2. **Set your environment variables (recommended):**
   Add an \`.env\` file with:
   - \`SPACE_API_URL=\` (target environment space URL)
   - \`OPENENV_API_KEY=\` (only if the environment space is protected)
   - \`HF_TOKEN=\` (only if your base model is gated/private)

3. **Run the experiment:**
   \`\`\`bash
   python main.py
   \`\`\`

4. **Optional API mode (no Gradio):**
   \`\`\`bash
   python app.py
   \`\`\`
   Then trigger training with:
   \`\`\`bash
   curl -X POST http://localhost:7860/train
   \`\`\`
   Fetch logs with:
   \`\`\`bash
   curl http://localhost:7860/logs
   \`\`\`

## What It Does
- Evaluates the "untrained" model on the hard task to establish a baseline.
- Trains the model on the easy and medium tasks using a trial-and-error Policy Gradient update (updating the neural network weights based on the reward score).
- Evaluates the "trained" model on the hard task again.
- Generates a graph (\`training_results.png\`) comparing performance!

## Hackathon-Aligned Workflow

Use this order to stay aligned with OpenEnv RL best practices:

1. Pick a task with objective verification
- Step-by-step agent behavior should be possible.
- Success must be programmatically checkable.
- Difficulty should allow non-zero success probability.

2. Stabilize environment before scaling
- Confirm `reset` and `step` behavior.
- Confirm done/timeout behavior.
- Confirm local and remote runs both work.

3. Train with verifier-first rewards
- This repo uses independent reward columns:
   - environment score,
   - schema validity,
   - taxonomy validity,
   - process validity,
   - repeated-action and drift penalties.

4. Monitor more than one scalar
- Watch overall reward and component means.
- Inspect suspicious behavior warnings in logs.
- Sample model generations periodically.

5. Scale only after reward quality is stable
- Increase episodes, batching, or task diversity only after verifier metrics are healthy.

## Reward Hacking Defenses Included

- Multiple independent reward functions (outcome + process checks).
- Repeated identical action penalties.
- Drift penalty when clause changes but action does not.
- Early episode stop when repeated action behavior exceeds hard threshold.
- Configurable inspection cadence and suspicious-step warnings.

## Key Environment Variables

- `REWARD_ENV_WEIGHT`, `REWARD_SCHEMA_BONUS`, `REWARD_TAXONOMY_BONUS`, `REWARD_PROCESS_BONUS`
- `REWARD_REPEAT_PENALTY`, `REWARD_DRIFT_PENALTY`
- `REWARD_MIN`, `REWARD_MAX`
- `MIN_REASONING_CHARS`, `MAX_REASONING_CHARS`, `MIN_GROUNDING_OVERLAP`
- `REPEATED_ACTION_SOFT_LIMIT`, `REPEATED_ACTION_HARD_LIMIT`
- `INSPECT_EVERY_N_STEPS`, `WARN_IF_SUSPICIOUS_STEPS`
- `OPENENV_TIMEOUT_SECONDS`, `EPISODE_TIMEOUT_SECONDS`
- `MAX_NEW_TOKENS` (default: 512 — must be ≥190 for valid completions)
- `GRPO_NUM_GENERATIONS`, `GRPO_GRAD_ACCUM_STEPS`, `GRPO_SAMPLES_PER_TASK`
- `CURRICULUM_UNLOCK_THRESHOLD`, `CURRICULUM_WINDOW`

## SFT vs RL Rule of Thumb

- Plenty of high-quality traces available: start with SFT.
- Little/no trace data but strong verifier available: use RL/RLVR.
- Best practical path: light SFT warm start, then RL improvement.
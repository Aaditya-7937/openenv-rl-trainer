---
title: OpenEnv RL Trainer
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: gradio
python_version: "3.10"
app_file: app.py
pinned: false
---

# OpenEnv RL Trainer

This folder contains a fully modular, PyTorch-first Reinforcement Learning (RL) training pipeline for the OpenEnv Contract Review benchmark.

## Architecture

The project is structured using clean Software Design Principles (Separation of Concerns, Dependency Injection, Modularity).

### Files:
- \`main.py\`: The plug-and-play entrypoint orchestrating the experiment.
- \`app.py\`: Lightweight FastAPI service to trigger training and fetch logs/results.
- \`src/config.py\`: Centralized hyperparameters and settings.
- \`src/env_client.py\`: Handles HTTP communication with the Hugging Face space.
- \`src/agent.py\`: The AI Agent containing the LLM, tokenization, and RL optimization logic (Policy Gradient).
- \`src/evaluation.py\`: Logic for running blind evaluations, collecting metrics, and plotting graphs.

## How to Run

1. **Install requirements:**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`
   *(Note: \`trl\` is removed to avoid version and import conflicts. This project uses vanilla PyTorch Policy Gradients to demonstrate the exact math of RL without dependency bloat.)*

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
import os
import textwrap
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, Tuple
from .config import RLConfig


class RLAgent:
    """
    An agent that implements a standard Policy Gradient (REINFORCE) algorithm
    to map textual observations to generative actions and update model weights.
    """

    def __init__(self, config: RLConfig):
        self.config = config
        self.device = config.device
        print(f"[Agent] Loading {config.model_name} onto {self.device}")

        # Optional: Auth token for gated models
        hf_token = os.getenv("HF_TOKEN")

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, token=hf_token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the base model in bfloat16 on GPU to save memory.
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
            "low_cpu_mem_usage": True,
            "token": hf_token,
        }
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"

        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **model_kwargs,
        )

        if self.device != "cuda":
            base_model.to(self.device)

        # Enable gradient checkpointing to reduce memory usage during RL updates.
        base_model.gradient_checkpointing_enable()
        base_model.config.use_cache = False

        try:
            from peft import get_peft_model, LoraConfig, TaskType

            print("[Agent] Applying LoRA (PEFT) to drastically reduce VRAM usage...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
            )
            self.model = get_peft_model(base_model, peft_config)
            self.model.print_trainable_parameters()
        except ImportError:
            print(
                "[Agent] 'peft' not installed. Falling back to full model training (Warning: High VRAM needed!)"
            )
            self.model = base_model

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate
        )

        self.analysis_template = textwrap.dedent(
            """\
        <analysis>
        clause_type=
        risk_level=
        flags=
        suggested_action=
        reasoning=
        </analysis>
        """
        ).strip()

    def create_prompt(self, obs: Dict[str, Any]) -> str:
        """Create a structured prompt from environment observations."""
        from .config import (
            VALID_CLAUSE_TYPES,
            VALID_RISK_LEVELS,
            VALID_SUGGESTED_ACTIONS,
        )

        clause = obs.get("clause_text", "")
        return (
            f"TASK: Classify this clause exactly according to the constraints.\n\n"
            f"CLAUSE: '{clause}'\n\n"
            f"Allowed clause_type: {', '.join(VALID_CLAUSE_TYPES)}\n"
            f"Allowed risk_level: {', '.join(VALID_RISK_LEVELS)}\n"
            f"Allowed suggested_action: {', '.join(VALID_SUGGESTED_ACTIONS)}\n\n"
            f"TEMPLATE:\n{self.analysis_template}"
        )

    def parse_action(self, generated_text: str) -> Dict[str, Any]:
        """Parse raw text into JSON action payload."""
        start_tag, end_tag = "<analysis>", "</analysis>"
        start = generated_text.find(start_tag)
        end = generated_text.find(end_tag)

        if start != -1 and end != -1:
            generated_text = generated_text[start + len(start_tag) : end]

        parsed = {}
        for line in generated_text.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                parsed[k.strip()] = v.strip()

        from .config import (
            VALID_CLAUSE_TYPES,
            VALID_RISK_LEVELS,
            VALID_SUGGESTED_ACTIONS,
        )

        c_type = parsed.get("clause_type", "confidentiality")
        r_level = parsed.get("risk_level", "low")
        s_action = parsed.get("suggested_action", "accept_as_is")

        # Fallback protections if the LLM hallucinated outside the taxonomy
        if c_type not in VALID_CLAUSE_TYPES:
            c_type = "confidentiality"
        if r_level not in VALID_RISK_LEVELS:
            r_level = "low"
        if s_action not in VALID_SUGGESTED_ACTIONS:
            s_action = "accept_as_is"

        return {
            "action_type": "classify",
            "clause_type": c_type,
            "risk_level": r_level,
            "flags": [],
            "suggested_action": s_action,
            "reasoning": parsed.get("reasoning", "No reasoning provided."),
        }

    def generate_and_get_logprobs(
        self, prompt: str
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        """
        Generate a text response while computing gradient-tracking log probabilities.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        self.model.train()

        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.train_do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.config.train_do_sample:
            generation_kwargs["temperature"] = self.config.train_temperature
            generation_kwargs["top_p"] = self.config.top_p

        # 1. Generate text (without gradients to save memory)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )

        # 2. Extract generated tokens
        full_sequence = output[0]
        prompt_length = input_ids.shape[1]
        generated_tokens = full_sequence[prompt_length:]

        # Parse the plain text action
        generated_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )
        action = self.parse_action(generated_text)

        # 3. Perform a forward pass WITH gradients across the full sequence
        # We calculate the log-probabilities of the generated tokens based on the prompt
        full_input_ids = full_sequence.unsqueeze(0)
        full_attention_mask = torch.ones_like(full_input_ids).to(self.device)

        forward_outputs = self.model(
            input_ids=full_input_ids, attention_mask=full_attention_mask
        )

        # 4. Extract logits specifically for the newly generated tokens
        # The logits at index `i` predict the token at index `i+1`
        logits = forward_outputs.logits[0, prompt_length - 1 : -1, :]

        if len(generated_tokens) == 0:
            return action, torch.tensor(0.0, device=self.device, requires_grad=True)

        # Calculate Log Probabilities WITH gradient tracking (grad_fn)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Gather the log prob of the specific token the model actually generated
        token_log_probs = log_probs.gather(
            dim=-1, index=generated_tokens.unsqueeze(-1)
        ).squeeze(-1)

        # Sum them up
        total_log_prob = token_log_probs.sum()

        return action, total_log_prob

    def update_model(self, log_prob: torch.Tensor, reward: float):
        """
        Update the model weights using the REINFORCE policy gradient mechanism.
        Formula: Loss = -log(pi(a|s)) * Reward
        """
        self.optimizer.zero_grad()

        # Negative sign, because PyTorch MINIMIZES loss, but we want to MAXIMIZE reward.
        reward_t = torch.tensor(float(reward), device=self.device, dtype=log_prob.dtype)
        loss = -log_prob * reward_t

        if not torch.isfinite(loss):
            print(f"[Agent] Skipping non-finite loss: {loss.item()}")
            return

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip_norm
        )
        self.optimizer.step()

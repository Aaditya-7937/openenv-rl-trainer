import os
import re
import textwrap
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import Dict, Any, Tuple, List
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

        # Override the model's built-in generation_config to suppress
        # "generation flags are not valid: ['top_p', 'top_k']" warnings.
        # Gemma models bake top_p/top_k into their generation_config.json,
        # which conflicts with our explicit settings.
        self.model.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )
        print("[Agent] Overrode model generation_config to prevent top_p/top_k conflicts.")

        self.baseline = 0.0
        self.baseline_alpha = 0.1
        self.entropy_coeff = 0.01  # Entropy bonus to encourage exploration

        # Track parse failure rate for monitoring
        self._parse_attempts = 0
        self._parse_failures = 0

    def create_prompt(self, obs: Dict[str, Any]) -> str:
        """
        Create a structured prompt from environment observations.
        Uses the tokenizer's chat template for instruction-tuned models.
        """
        from .config import (
            VALID_CLAUSE_TYPES,
            VALID_RISK_LEVELS,
            VALID_SUGGESTED_ACTIONS,
        )

        clause = obs.get("clause_text", "")
        contract_type = obs.get("contract_type", "Unknown")
        jurisdiction = obs.get("jurisdiction", "Unknown")
        parties = obs.get("parties", [])
        corrective_feedback = obs.get("corrective_feedback", "")
        last_feedback = obs.get("last_action_feedback", "")
        clause_idx = obs.get("clause_index", 0)
        total_clauses = obs.get("total_clauses", 1)

        system_msg = (
            "You are an expert legal contract reviewer. "
            "You must classify contract clauses precisely using the given taxonomy. "
            "Read the clause carefully and identify its TRUE legal nature — "
            "do NOT default to 'confidentiality' for every clause."
        )

        user_msg = (
            f"## Contract Context\n"
            f"- Type: {contract_type}\n"
            f"- Jurisdiction: {jurisdiction}\n"
            f"- Parties: {', '.join(parties) if parties else 'N/A'}\n"
            f"- Clause {clause_idx + 1} of {total_clauses}\n\n"
            f"## Clause to Classify\n"
            f'"{clause}"\n\n'
        )

        if corrective_feedback:
            user_msg += f"## Feedback from Previous Step\n{corrective_feedback}\n\n"
        if last_feedback:
            user_msg += f"## Environment Feedback\n{last_feedback}\n\n"

        user_msg += (
            f"## Allowed Values\n"
            f"clause_type: {', '.join(VALID_CLAUSE_TYPES)}\n"
            f"risk_level: {', '.join(VALID_RISK_LEVELS)}\n"
            f"suggested_action: {', '.join(VALID_SUGGESTED_ACTIONS)}\n\n"
            f"## Instructions\n"
            f"Classify this clause by filling in the template below. "
            f"Think carefully about what TYPE of clause this is based on its content. "
            f"For example:\n"
            f"- Clauses about damages/liability caps → limitation_of_liability\n"
            f"- Clauses about indemnify/hold harmless → indemnification\n"
            f"- Clauses about term/renewal/termination → termination\n"
            f"- Clauses about governing law → governing_law\n"
            f"- Clauses about insurance coverage → insurance\n"
            f"- Clauses about representations/warranties → representations or warranty\n"
            f"- Clauses about force majeure → force_majeure\n"
            f"- Clauses about assignment → assignment\n"
            f"- Clauses about confidential information → confidentiality\n\n"
            f"Respond ONLY with the filled template, nothing else:\n\n"
            f"<analysis>\n"
            f"clause_type=YOUR_ANSWER\n"
            f"risk_level=YOUR_ANSWER\n"
            f"flags=comma_separated_flags_or_empty\n"
            f"suggested_action=YOUR_ANSWER\n"
            f"reasoning=brief explanation of why you chose this classification\n"
            f"</analysis>"
        )

        # Use chat template if available (critical for instruction-tuned models)
        messages = [
            {"role": "user", "content": f"{system_msg}\n\n{user_msg}"},
        ]

        try:
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return formatted
        except Exception:
            # Fallback for models without chat template
            return f"{system_msg}\n\n{user_msg}"

    def parse_action(self, generated_text: str) -> Dict[str, Any]:
        """Parse raw text into JSON action payload with detailed logging on failures."""
        self._parse_attempts += 1

        from .config import (
            VALID_CLAUSE_TYPES,
            VALID_RISK_LEVELS,
            VALID_SUGGESTED_ACTIONS,
        )

        # Try to extract the <analysis>...</analysis> block
        start_tag, end_tag = "<analysis>", "</analysis>"
        start = generated_text.find(start_tag)
        end = generated_text.find(end_tag)

        parse_region = generated_text
        if start != -1 and end != -1:
            parse_region = generated_text[start + len(start_tag) : end]

        # Parse key=value pairs
        parsed = {}
        for line in parse_region.splitlines():
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip().lower()
                v = v.strip().lower()
                # Clean up common model quirks
                v = v.strip("'\"` ")
                parsed[k] = v

        # Also try regex for more flexible parsing (handles "clause_type: value" format)
        if "clause_type" not in parsed:
            for pattern_key in ["clause_type", "risk_level", "suggested_action"]:
                match = re.search(
                    rf"{pattern_key}\s*[=:]\s*['\"]?(\w+)['\"]?",
                    generated_text,
                    re.IGNORECASE,
                )
                if match and pattern_key not in parsed:
                    parsed[pattern_key] = match.group(1).lower()

        c_type = parsed.get("clause_type", "")
        r_level = parsed.get("risk_level", "")
        s_action = parsed.get("suggested_action", "")

        # Track if we had to use defaults (indicates parse failure)
        used_default = False

        if c_type not in VALID_CLAUSE_TYPES:
            # Try fuzzy matching before giving up
            c_type_matched = self._fuzzy_match(c_type, VALID_CLAUSE_TYPES)
            if c_type_matched:
                c_type = c_type_matched
            else:
                if c_type:
                    print(f"  [Parse] Unknown clause_type '{c_type}', defaulting")
                else:
                    print(f"  [Parse] Missing clause_type, defaulting")
                c_type = "confidentiality"
                used_default = True

        if r_level not in VALID_RISK_LEVELS:
            r_level_matched = self._fuzzy_match(r_level, VALID_RISK_LEVELS)
            if r_level_matched:
                r_level = r_level_matched
            else:
                if r_level:
                    print(f"  [Parse] Unknown risk_level '{r_level}', defaulting")
                r_level = "low"
                used_default = True

        if s_action not in VALID_SUGGESTED_ACTIONS:
            s_action_matched = self._fuzzy_match(s_action, VALID_SUGGESTED_ACTIONS)
            if s_action_matched:
                s_action = s_action_matched
            else:
                if s_action:
                    print(f"  [Parse] Unknown suggested_action '{s_action}', defaulting")
                s_action = "accept_as_is"
                used_default = True

        if used_default:
            self._parse_failures += 1
            if self._parse_attempts % 10 == 0:
                rate = self._parse_failures / self._parse_attempts * 100
                print(
                    f"  [Parse Stats] {self._parse_failures}/{self._parse_attempts} "
                    f"parse failures ({rate:.0f}%)"
                )

        reasoning = parsed.get("reasoning", "No reasoning provided.")

        return {
            "action_type": "classify",
            "clause_type": c_type,
            "risk_level": r_level,
            "flags": [],
            "suggested_action": s_action,
            "reasoning": reasoning,
        }

    @staticmethod
    def _fuzzy_match(value: str, valid_options: list) -> str | None:
        """Try to fuzzy-match a value against valid options."""
        if not value:
            return None
        value = value.lower().strip()
        # Direct substring match
        for opt in valid_options:
            if value in opt or opt in value:
                return opt
        # Try replacing common separators
        normalized = value.replace(" ", "_").replace("-", "_")
        if normalized in valid_options:
            return normalized
        return None

    def generate_and_get_logprobs(
        self, prompt: str
    ) -> Tuple[Dict[str, Any], torch.Tensor, str]:
        """
        Generate a text response while computing gradient-tracking log probabilities.
        Returns (action_dict, log_prob, generated_text)
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

        num_generated = len(generated_tokens)
        if num_generated == 0:
            return action, torch.tensor(0.0, device=self.device, requires_grad=True), generated_text

        # Calculate Log Probabilities WITH gradient tracking (grad_fn)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Gather the log prob of the specific token the model actually generated
        token_log_probs = log_probs.gather(
            dim=-1, index=generated_tokens.unsqueeze(-1)
        ).squeeze(-1)

        # Use MEAN log-prob instead of SUM to normalize by sequence length.
        # Without this, long outputs get massive log-probs that dominate
        # the gradient and cause instability.
        mean_log_prob = token_log_probs.sum() / num_generated

        # Calculate entropy bonus (encourages diverse token predictions)
        # Higher entropy = model is less certain = more exploration
        probs = torch.nn.functional.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        self._last_entropy = entropy  # Save for use in update_model

        return action, mean_log_prob, generated_text

    def update_model(self, log_prob: torch.Tensor, reward: float):
        """
        Update the model weights using the REINFORCE policy gradient mechanism.
        Formula: Loss = -log(pi(a|s)) * (Reward - Baseline) - entropy_coeff * H(pi)
        """
        self.optimizer.zero_grad()

        # Calculate advantage
        advantage = reward - self.baseline
        
        # Update baseline (moving average)
        self.baseline = (1 - self.baseline_alpha) * self.baseline + self.baseline_alpha * reward

        # Negative sign, because PyTorch MINIMIZES loss, but we want to MAXIMIZE advantage.
        advantage_t = torch.tensor(float(advantage), device=self.device, dtype=log_prob.dtype)
        policy_loss = -log_prob * advantage_t

        # Add entropy bonus: subtract entropy to encourage exploration
        # (we subtract because we MINIMIZE loss, and we WANT higher entropy)
        entropy = getattr(self, '_last_entropy', torch.tensor(0.0, device=self.device))
        loss = policy_loss - self.entropy_coeff * entropy

        if not torch.isfinite(loss):
            print(f"[Agent] Skipping non-finite loss: {loss.item()}")
            return

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip_norm
        )
        self.optimizer.step()

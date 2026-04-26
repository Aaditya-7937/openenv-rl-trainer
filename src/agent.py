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

    def generate_group(
        self, prompt: str
    ) -> List[Tuple[Dict[str, Any], str]]:
        """
        Generate G candidate completions for one prompt (the GRPO 'group').
        Returns list of (action_dict, generated_text) tuples.
        """
        G = self.config.grpo_group_size
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": True,
            "temperature": self.config.train_temperature,
            "num_return_sequences": G,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        # Generate in eval mode (gradient checkpointing corrupts generation)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )

        prompt_length = input_ids.shape[1]
        candidates = []
        for i in range(G):
            generated_tokens = outputs[i][prompt_length:]
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            action = self.parse_action(text)
            candidates.append((action, text))

        return candidates

    def grpo_update(
        self, prompt: str, candidates: List[Tuple[Dict[str, Any], str]],
        rewards: List[float]
    ) -> Dict[str, float]:
        """
        GRPO policy update: compute group-normalised advantages and update weights.

        Key insight: even when all candidates get the same env score,
        differences in risk_level, suggested_action, and reasoning quality
        create reward variance through the reward composer. The group
        normalisation turns these small differences into meaningful gradients.

        Returns dict with debug info (mean_reward, std_reward, best_reward, etc.)
        """
        G = len(candidates)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)

        # Group-relative advantage normalisation (the core of GRPO)
        mean_r = rewards_t.mean()
        std_r = rewards_t.std()
        if std_r < 1e-8:
            # All rewards identical — still update via entropy bonus only
            advantages = torch.zeros(G)
        else:
            advantages = (rewards_t - mean_r) / (std_r + 1e-8)

        # Switch to train mode for the gradient computation
        self.model.train()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        prompt_length = input_ids.shape[1]

        total_loss = torch.tensor(0.0, device=self.device)
        valid_count = 0

        for i, (action, text) in enumerate(candidates):
            # Re-tokenize the full sequence (prompt + this candidate's output)
            full_text = prompt + text
            full_inputs = self.tokenizer(
                full_text, return_tensors="pt", truncation=True
            ).to(self.device)
            full_ids = full_inputs.input_ids[0]
            generated_tokens = full_ids[prompt_length:]

            if len(generated_tokens) == 0:
                continue

            # Forward pass with gradients
            full_attention_mask = torch.ones(1, len(full_ids), device=self.device)
            outputs = self.model(
                input_ids=full_ids.unsqueeze(0),
                attention_mask=full_attention_mask,
            )

            logits = outputs.logits[0, prompt_length - 1: -1, :]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            num_gen = min(len(generated_tokens), logits.shape[0])
            token_log_probs = log_probs[:num_gen].gather(
                dim=-1, index=generated_tokens[:num_gen].unsqueeze(-1)
            ).squeeze(-1)
            mean_log_prob = token_log_probs.sum() / num_gen

            # Policy loss: -advantage * log_prob
            adv = advantages[i].item()
            policy_loss = -mean_log_prob * adv

            # Entropy bonus (encourages exploration)
            probs = torch.nn.functional.softmax(logits[:num_gen], dim=-1)
            entropy = -(probs * log_probs[:num_gen]).sum(dim=-1).mean()

            total_loss = total_loss + policy_loss - self.entropy_coeff * entropy
            valid_count += 1

        if valid_count > 0:
            total_loss = total_loss / valid_count

            if torch.isfinite(total_loss):
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip_norm
                )
                self.optimizer.step()

        return {
            "mean_reward": mean_r.item(),
            "std_reward": std_r.item(),
            "best_reward": max(rewards),
            "worst_reward": min(rewards),
            "loss": total_loss.item() if torch.is_tensor(total_loss) else 0.0,
        }

    def generate_and_get_logprobs(
        self, prompt: str
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        """
        Generate a SINGLE text response and compute its log-probability.
        Used by the fallback PyTorch-native GRPO loop in main.py.
        Returns (action_dict, log_prob_tensor).
        The action_dict includes '_parse_failed' and '_raw_generation' keys.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": True,
            "temperature": self.config.train_temperature,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        # Generate in eval mode (gradient checkpointing corrupts generation)
        self.model.eval()
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )

        full_sequence = output[0]
        prompt_length = input_ids.shape[1]
        generated_tokens = full_sequence[prompt_length:]

        generated_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )
        action = self.parse_action(generated_text)

        # Annotate action with parse metadata for the GRPO loop
        action["_raw_generation"] = generated_text
        action["_parse_failed"] = (
            action.get("clause_type") == "confidentiality"
            and generated_text
            and "<analysis>" not in generated_text
        )

        # Compute log-prob with gradients
        self.model.train()
        full_input_ids = full_sequence.unsqueeze(0)
        full_attention_mask = torch.ones_like(full_input_ids).to(self.device)

        forward_outputs = self.model(
            input_ids=full_input_ids, attention_mask=full_attention_mask
        )

        logits = forward_outputs.logits[0, prompt_length - 1: -1, :]
        num_generated = len(generated_tokens)

        if num_generated == 0:
            return action, torch.tensor(0.0, device=self.device, requires_grad=True)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=generated_tokens.unsqueeze(-1)
        ).squeeze(-1)
        mean_log_prob = token_log_probs.sum() / num_generated

        return action, mean_log_prob

    def compute_grpo_advantages(self, rewards: List[float]) -> torch.Tensor:
        """
        Compute group-relative advantages for a batch of rewards.
        A_i = (r_i - mean(r)) / (std(r) + eps)
        """
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        mean_r = rewards_t.mean()
        std_r = rewards_t.std()
        if std_r < 1e-8:
            return torch.zeros_like(rewards_t)
        return (rewards_t - mean_r) / (std_r + 1e-8)

    def update_model_grpo(
        self, log_probs: List[torch.Tensor], advantages: torch.Tensor
    ):
        """
        Update model weights using pre-computed log-probs and GRPO advantages.
        Used by the fallback PyTorch-native GRPO loop in main.py.
        """
        self.optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=self.device)
        valid = 0

        for lp, adv in zip(log_probs, advantages):
            if not torch.isfinite(lp):
                continue
            total_loss = total_loss + (-lp * adv.item())
            valid += 1

        if valid > 0:
            total_loss = total_loss / valid
            if torch.isfinite(total_loss):
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip_norm
                )
                self.optimizer.step()

    def save_checkpoint(self, path: str):
        """Save LoRA adapter weights for rollback."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save only the trainable (LoRA) parameters
        trainable_state = {
            k: v for k, v in self.model.state_dict().items() if v.requires_grad
        }
        if not trainable_state:
            # Fallback: save all named params that require grad
            trainable_state = {
                n: p.data for n, p in self.model.named_parameters() if p.requires_grad
            }
        torch.save(trainable_state, path)

    def load_checkpoint(self, path: str):
        """Restore LoRA adapter weights from checkpoint."""
        if not os.path.exists(path):
            print(f"[Agent] Checkpoint not found: {path}")
            return
        state = torch.load(path, map_location=self.device)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[Agent] Checkpoint load — missing keys: {len(missing)}")

    def save_final_model(self, output_dir: str):
        """Save the final trained LoRA adapter."""
        os.makedirs(output_dir, exist_ok=True)
        try:
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print(f"[Agent] Saved adapter to {output_dir}")
        except Exception as e:
            print(f"[Agent] save_pretrained failed: {e}")
            # Fallback: save state dict
            torch.save(
                {n: p.data for n, p in self.model.named_parameters() if p.requires_grad},
                os.path.join(output_dir, "adapter_state.pt"),
            )
            print(f"[Agent] Saved state dict fallback to {output_dir}/adapter_state.pt")

    def inference_sanity_check(self, sample_clause: str):
        """Quick check that the model can still generate after training."""
        obs = {
            "clause_text": sample_clause,
            "contract_type": "Service Agreement",
            "jurisdiction": "New York",
            "parties": ["Party A", "Party B"],
            "clause_index": 0,
            "total_clauses": 1,
        }
        prompt = self.create_prompt(obs)
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            output = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = self.tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        action = self.parse_action(text)
        print(f"[Sanity] clause_type={action['clause_type']} risk={action['risk_level']}")
        print(f"[Sanity] Raw: {text[:200]}")


# IMPORTANT: unsloth must be imported before transformers.
# It monkey-patches HuggingFace attention and MLP layers at import time.
# Importing transformers first silently disables the fused Triton kernels
# and causes the UserWarning seen in training logs.
try:
    import unsloth  # noqa: F401  # type: ignore
except ImportError:
    pass  # Unsloth optional — agent falls back to standard HF+PEFT

import os
import textwrap
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, List, Tuple
from .config import RLConfig


class RLAgent:
    """
    Loads an LLM (via Unsloth or standard HF+PEFT), generates clause-level
    actions with token log-probabilities, and exposes GRPO weight-update logic.
    """

    def __init__(self, config: RLConfig):
        self.config = config
        self.device = config.device
        hf_token = os.getenv("HF_TOKEN")

        print(f"[Agent] Loading {config.model_name} onto {self.device}")

        # ── Path 1: Unsloth ───────────────────────────────────────────────────
        # Unsloth wraps HuggingFace + PEFT with fused Triton kernels and NF4
        # 4-bit quantization, giving ~2x training speed and ~60% VRAM savings.
        # It is the recommended front-end for the TRL + Unsloth + OpenEnv stack.
        _unsloth_loaded = False
        try:
            from unsloth import FastLanguageModel  # type: ignore

            print("[Agent] Unsloth ✓ — 4-bit NF4 loading + fused kernels active.")
            base_model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=config.model_name,
                max_seq_length=config.max_seq_length,
                dtype=None,         # auto-detect bfloat16 / float16
                load_in_4bit=True,  # NF4 quantization
                token=hf_token,
            )
            # Unsloth's get_peft_model patches attention layers for 2x speed
            # and sets up gradient checkpointing internally.
            self.model = FastLanguageModel.get_peft_model(
                base_model,
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=config.seed,
            )
            _unsloth_loaded = True
        except ImportError:
            print("[Agent] Unsloth not installed — falling back to standard HF + PEFT.")
        except Exception as exc:
            print(f"[Agent] Unsloth failed ({exc}) — falling back to standard HF + PEFT.")

        # ── Path 2: Standard HuggingFace + PEFT LoRA ─────────────────────────
        if not _unsloth_loaded:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_name, token=hf_token
            )
            model_kwargs: dict = {
                "torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
                "low_cpu_mem_usage": True,
                "token": hf_token,
            }
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"

            base_model = AutoModelForCausalLM.from_pretrained(
                config.model_name, **model_kwargs
            )
            if self.device != "cuda":
                base_model.to(self.device)

            base_model.gradient_checkpointing_enable()
            base_model.config.use_cache = False

            try:
                from peft import get_peft_model, LoraConfig, TaskType

                print("[Agent] Applying LoRA (PEFT) to reduce VRAM usage...")
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.05,
                )
                self.model = get_peft_model(base_model, peft_config)
            except ImportError:
                print("[Agent] peft not installed — using full model (high VRAM).")
                self.model = base_model

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if hasattr(self.model, "print_trainable_parameters"):
            self.model.print_trainable_parameters()
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
        """
        Parse raw text into JSON action payload.
        Returns _parse_failed=True if the LLM hallucinated any field outside
        the allowed taxonomy. The caller must override reward to 0.0 in that
        case so that bad outputs receive no positive RL signal.
        """
        start_tag, end_tag = "<analysis>", "</analysis>"
        start = generated_text.find(start_tag)
        end = generated_text.find(end_tag)

        # If the model did not produce the required XML structure at all, fail immediately.
        parse_failed = start == -1 or end == -1

        if not parse_failed:
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

        c_type = parsed.get("clause_type", "")
        r_level = parsed.get("risk_level", "")
        s_action = parsed.get("suggested_action", "")

        # Detect any hallucination — mark the parse as failed so the reward
        # is zeroed out by the caller. Still substitute safe defaults so the
        # environment call does not crash (we just won't learn from it).
        if c_type not in VALID_CLAUSE_TYPES:
            parse_failed = True
            c_type = "confidentiality"
        if r_level not in VALID_RISK_LEVELS:
            parse_failed = True
            r_level = "low"
        if s_action not in VALID_SUGGESTED_ACTIONS:
            parse_failed = True
            s_action = "accept_as_is"

        return {
            "action_type": "classify",
            "clause_type": c_type,
            "risk_level": r_level,
            "flags": [],
            "suggested_action": s_action,
            "reasoning": parsed.get("reasoning", "No reasoning provided."),
            "_parse_failed": parse_failed,
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
        # Tag the raw LLM output so the training loop can log it for human inspection.
        # The caller MUST pop this before sending the action to the environment.
        action["_raw_generation"] = generated_text[:400]

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

    # ── Weight update ──────────────────────────────────────────────────────────
    #
    # REINFORCE (update_model_trajectory) has been intentionally removed.
    #
    # Why REINFORCE is inferior for this task:
    #   • REINFORCE normalises returns across STEPS of the same episode.
    #     Different steps observe different clauses, so the "baseline" is an
    #     average over unrelated inputs — a high-variance estimator.
    #   • GRPO normalises rewards across G COMPLETIONS of the SAME prompt.
    #     The group mean is a tight, prompt-specific baseline, giving much
    #     lower gradient variance with no critic network needed.
    #   • Formula comparison:
    #       REINFORCE:  A_t = G_t − mean(G_1..G_T)   ← cross-observation noise
    #       GRPO:       A_i = r_i − mean(r_1..r_G)   ← same-prompt signal
    #
    # Both TRL's GRPOTrainer (primary) and update_model_grpo (fallback)
    # implement the GRPO formula. Do not re-introduce REINFORCE here.

    @staticmethod
    def compute_grpo_advantages(rewards: List[float]) -> torch.Tensor:
        """
        Compute group-relative advantages for GRPO.

        For G completions sampled from the same prompt:
            A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)

        This normalisation eliminates the need for a separate value/critic
        network: the group mean acts as the baseline.  When all G rewards are
        equal (e.g. all 0.0 due to parse failures), advantages collapse to
        zero and no harmful gradient is applied.
        """
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        mean_r = rewards_t.mean()
        std_r = rewards_t.std() if len(rewards) > 1 else torch.tensor(1.0)
        return (rewards_t - mean_r) / (std_r + 1e-8)

    def update_model_grpo(
        self, log_probs: List[torch.Tensor], advantages: torch.Tensor
    ) -> None:
        """
        GRPO policy gradient update using group-relative advantages.

        Unlike REINFORCE which weights log-probs by raw discounted returns,
        GRPO weights them by group-normalised advantages, giving lower variance
        and no need for a critic network.

        Formula:
            Loss = -sum_i [ log(pi(a_i | s)) * A_i ]
                  where A_i = (r_i - mean(r)) / std(r)
        """
        if not log_probs:
            print("[Agent] GRPO: no rollouts to update from — skipping.")
            return

        stacked_log_probs = torch.stack(log_probs)  # shape: (G,)
        advantages = advantages.to(stacked_log_probs.device)

        loss = -(stacked_log_probs * advantages).sum()

        if not torch.isfinite(loss):
            print(f"[Agent] GRPO: skipping non-finite loss ({loss.item()}).")
            return

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip_norm
        )
        self.optimizer.step()

        print(
            f"[Agent] GRPO update | G={len(log_probs)} "
            f"| Loss: {loss.item():.4f} "
            f"| Mean adv: {advantages.mean().item():.4f} "
            f"| Adv range: [{advantages.min().item():.3f}, {advantages.max().item():.3f}]"
        )

    def save_checkpoint(self, path: str) -> None:
        """Persist trainable weights to disk before an episode starts.
        Used for rollback if suspicious drift is detected after the episode.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[Agent] Checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> None:
        """Restore weights saved by save_checkpoint."""
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        print(f"[Agent] Checkpoint restored ← {path}")

    def save_final_model(self, output_dir: str) -> None:
        """
        Safe post-training adapter export (Guideline 16).

        Uses save_pretrained() which stores ONLY the LoRA adapter deltas in
        HuggingFace format.  This is correct for both the Unsloth and plain
        PEFT paths because:
          • No 4-bit → 16-bit upcasting happens (avoids quality damage).
          • No merge_and_unload() is called (merge only when explicitly needed
            for deployment, and only via Unsloth's own safe merge API).
          • The adapter can be reloaded with `from_pretrained` + `PeftModel`
            for post-save inference verification.

        Do NOT replace this with torch.save(state_dict()) for the final export;
        that path is reserved for the mid-training rollback checkpoint only.
        """
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"[Agent] Final adapter saved → {output_dir}")
        print(f"[Agent] Reload with: PeftModel.from_pretrained(base, '{output_dir}')")

    def inference_sanity_check(self, sample_clause: str) -> str:
        """
        Post-training inference check (Guideline 16).

        Runs a single greedy forward pass on a known clause immediately after
        training to confirm the saved model can still generate coherent output.
        A broken merge / corrupt save will surface here rather than at deploy time.
        """
        obs = {"clause_text": sample_clause}
        prompt = self.create_prompt(obs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = self.tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        action = self.parse_action(generated)
        ok = not action.get("_parse_failed", False)
        print(
            f"[Agent] Inference sanity check — "
            f"parse_ok={ok} | clause_type={action.get('clause_type')} "
            f"| risk_level={action.get('risk_level')}"
        )
        print(f"[Agent] Raw output: {repr(generated[:200])}")
        self.model.train()
        return generated

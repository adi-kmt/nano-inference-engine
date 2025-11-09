# engine/speculative_engine.py
from typing import Optional, Iterator, List, Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoModelForCausalLM, TextIteratorStreamer, \
    DynamicCache

from config import SpeculativeDecodingParams
from .engine import Engine, SamplingParams


"""
TODOs
1. handle attention masks 
2. use cache across the generations
"""



class SpeculativeEngine(Engine):
    def __init__(
        self,
    ):
        self.draft_model = None
        super().__init__()

        self.draft_config = SpeculativeDecodingParams()
        self.draft_model_cache = DynamicCache()

    def load_draft_model(self, draft_model_name: str, sampling_config: SpeculativeDecodingParams):
        """Load a small, fast draft model (e.g., TinyLlama, Phi-2, etc.)"""
        print(f"Loading draft model '{draft_model_name}'...")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "auto":
            self.draft_model.to(self.device)
        self.draft_model.eval()
        print("✅ Draft model loaded!")

    def _draft_and_verify(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        assert input_ids.shape[0] == 1, "Batch size must be 1 for speculative decoding"
        prompt_len = input_ids.shape[1]

        draft_input_ids = input_ids.clone()
        draft_probs = []
        draft_tokens = []

        #Run all the draft input ids sequentially
        with torch.no_grad():
            for _ in range(self.draft_config.num_speculative_tokens):
                outputs = self.draft_model(draft_input_ids, attention_mask=attention_mask)
                output_logits = outputs.logits[:, -1, :]
                output_probs = torch.softmax(output_logits, dim=-1)
                next_token = torch.multinomial(output_probs, num_samples=1).item()
                draft_tokens.append(next_token)
                draft_input_ids = torch.cat([draft_input_ids, torch.tensor([[next_token]])], dim=-1)
                draft_probs.append(output_probs[0])

        # TArget verification
        with torch.no_grad():
            target_outputs = self.model(draft_input_ids)
            target_logits = target_outputs.logits[:,
                            prompt_len - 1:prompt_len - 1 + self.draft_config.num_speculative_tokens, :]
            target_probs = torch.softmax(target_logits, dim=-1)

        accepted_draft_tokens = []
        all_final_tokens = []

        for i, token_id in enumerate(draft_tokens):
            draft_prob = draft_probs[i][token_id]
            target_prob = target_probs[0, i, token_id]

            if draft_prob <= 1e-15:
                accept_prob = 0.0
            else:
                accept_prob = min(1.0, target_prob / draft_prob)

            if torch.rand(1).item() < accept_prob:
                # Draft token accepted
                accepted_draft_tokens.append(token_id)
                all_final_tokens.append(token_id)
            else:
                # Draft token rejected - sample correction
                adjusted_dist = torch.clamp(target_probs[0, i] - accept_prob * draft_probs[i], min=0.0)
                # If the adjusted distribution doens't sum to 0, then sample from it
                if adjusted_dist.sum() > 0:
                    adjusted_dist = adjusted_dist / adjusted_dist.sum()
                    corrected_token = torch.multinomial(adjusted_dist, num_samples=1).item()
                    all_final_tokens.append(corrected_token)
                else:
                    corrected_token = torch.multinomial(target_probs[0, i], 1).item()
                    all_final_tokens.append(corrected_token)
                break

        # When all draft tokens accepted - generate the next token
        if len(accepted_draft_tokens) == len(draft_tokens):
            final_input = torch.cat([input_ids, torch.tensor([all_final_tokens], device=input_ids.device)], dim=1)
            with torch.no_grad():
                final_logits = self.model(final_input).logits[:, -1, :]
                final_probs = torch.softmax(final_logits, dim=-1)
                extra_token = torch.multinomial(final_probs, 1).item()
                all_final_tokens.append(extra_token)


        new_tokens = torch.tensor([all_final_tokens], device=input_ids.device)
        updated_input_ids = torch.cat([input_ids, new_tokens], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, len(all_final_tokens)), dtype=torch.long, device=input_ids.device)], dim=1)

        return updated_input_ids, attention_mask, len(all_final_tokens)

    def generate(self, prompt: str) -> str:
        """Non-streaming generation with speculative decoding."""
        if self.model is None or self.tokenizer is None or self.draft_model is None:
            raise RuntimeError("Target model, draft model, and tokenizer must be loaded.")

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        max_new_tokens = getattr(self.params, 'max_new_tokens', 100)

        all_new_tokens = []
        with torch.no_grad():
            generated_ids = input_ids.clone()
            current_attention_mask = attention_mask.clone()
            total_generated = 0

            while total_generated < max_new_tokens:
                # Run one speculative step
                new_input_ids, new_attention_mask, n_accepted = self._draft_and_verify(
                    generated_ids, current_attention_mask
                )

                # Extract only the newly accepted tokens
                newly_added = new_input_ids[:, generated_ids.shape[1]:]

                # Update the next iteration
                generated_ids = new_input_ids
                current_attention_mask = new_attention_mask
                total_generated += n_accepted

                # Collectall new tokens
                if n_accepted > 0:
                    all_new_tokens.append(newly_added[0])

                if (newly_added == self.tokenizer.eos_token_id).any():
                    break

                if total_generated >= max_new_tokens:
                    break

        # Combine all tokens and decode
        if all_new_tokens:
            final_tokens = torch.cat(all_new_tokens, dim=0)
            return self.tokenizer.decode(final_tokens, skip_special_tokens=True)
        return ""
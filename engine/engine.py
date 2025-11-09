# engine/engine.py
from typing import Optional, Iterator
import torch
from config import SamplingParams
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids: list):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0][-1].item() in self.stop_token_ids


class Engine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.params = SamplingParams()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._compiled = False

    def load_model(self, model_name: str):
        """Load model with proper device and dtype handling"""
        print(f"Loading model on {self.device} with dtype {self.dtype}...")

        # Load in appropriate dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.model.eval()

        # Optional: compile for speed (may break streaming in some PyTorch versions)
        # If you encounter issues, comment out compilation or use mode="default"
        try:
            print("Compiling model...")
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)
            self._compiled = True
        except Exception as e:
            print(f"⚠️  Compilation failed (falling back to eager): {e}")
            self._compiled = False

        print("✅ Model loaded successfully!")

    def _prepare_generation_kwargs(self, inputs: dict) -> dict:
        """Centralize generation config"""
        kwargs = {
            "max_new_tokens": self.params.max_new_tokens,
            "do_sample": self.params.temperature > 0,
            "temperature": self.params.temperature,
            "top_p": self.params.top_p,
            "top_k": self.params.top_k,
            "repetition_penalty": self.params.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Add custom stopping criteria if needed
        if self.tokenizer.eos_token_id is not None:
            kwargs["stopping_criteria"] = StoppingCriteriaList([
                StopOnTokens([self.tokenizer.eos_token_id])
            ])

        return kwargs

    def generate(self, prompt: str) -> str:
        """Original API: full string generation (backward compatible)"""
        gen_out = self.generate_stream(prompt)
        full_text = ""
        for text in gen_out:
            full_text += text
        return full_text

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """NEW: Accurate streaming generation for benchmarking & real apps"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        # Prepare streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=30.0
        )

        # Build generation kwargs
        gen_kwargs = self._prepare_generation_kwargs(inputs)
        gen_kwargs.update({
            **inputs,
            "streamer": streamer,
        })

        # Run generation in background
        import threading
        def run_generate():
            with torch.no_grad():
                self.model.generate(**gen_kwargs)

        thread = threading.Thread(target=run_generate, daemon=True)
        thread.start()

        # Yield tokens as they arrive
        try:
            for new_text in streamer:
                yield new_text
        except Exception as e:
            thread.join(timeout=1)
            raise RuntimeError(f"Generation failed: {e}") from e
        finally:
            thread.join(timeout=5)
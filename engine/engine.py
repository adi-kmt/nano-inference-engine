from typing import Optional

import torch

from config import SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache, AutoConfig


class Engine:
    def __init__(self):
        self.kv_cache: Optional[StaticCache] = None
        self.model = None
        self.tokenizer = None
        self.params = SamplingParams()


    def load_model(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True
        )
        self.kv_cache = StaticCache(
            config=AutoConfig.from_pretrained(model_name),
            max_cache_len=2048
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model = torch.compile(self.model)
        self.model.eval()


    def generate(self, prompt:str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")

        print("🔍 Before generation:")
        print(f"  Cache value[0] sum: {self.kv_cache.layers[0].values}")

        outputs = self.model.generate(
            past_key_values=self.kv_cache,
            use_cache=True,
            **inputs,
            **self.params.model_dump()
        )

        print("🔍 After generation:")
        print(f"  Cache value[0] sum: {self.kv_cache.layers[0].values}")# 👈 important!

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
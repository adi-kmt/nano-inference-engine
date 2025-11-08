from typing import Optional

from config import SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer


class Engine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.params: Optional[SamplingParams] = None


    def load_model(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model.eval()


    def generate(self, prompt:str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
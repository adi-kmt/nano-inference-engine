from pydantic import BaseModel

class SamplingParams(BaseModel):
    temperature: int = 0
    top_p: int = 0.6
    top_k: int = 0.9
    max_tokens: int = 512
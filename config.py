from pydantic import BaseModel

class SamplingParams(BaseModel):
    temperature: float = 0.5
    top_p: float = 0.6
    top_k: int = 50
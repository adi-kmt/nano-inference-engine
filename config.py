from pydantic import BaseModel

class SamplingParams(BaseModel):
    temperature: float = 0.5
    top_p: float = 0.6
    top_k: int = 50
    max_new_tokens: int = 1024
    repetition_penalty: float = 1

class SpeculativeDecodingParams(BaseModel):
    sampling_params = SamplingParams()
    num_speculative_tokens: int = 128
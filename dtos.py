from pydantic import BaseModel


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    stop: list = None

class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list
    usage: dict
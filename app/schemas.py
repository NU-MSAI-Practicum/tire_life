from pydantic import BaseModel
from typing import List

class InitialState(BaseModel):
    initial_state: List[float]

class PredictionResponse(BaseModel):
    State: List[float]
    Action: List[int]
    NextState: List[float]
    Reward: float
    Log: str

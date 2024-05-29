# app/models.py
from pydantic import BaseModel
from typing import List

class PredictionInput(BaseModel):
    num_trucks: int
    num_tires_per_truck: int
    initial_state: List[List[float]]
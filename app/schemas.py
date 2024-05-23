# app/schemas.py

from pydantic import BaseModel, Field, ValidationError, validator
from typing import List, Any

class Matrix(BaseModel):
    data: List[List[float]]  # List of lists, where each sublist is a row

    @validator('data')
    def check_data_shape(cls, value):
        if len(value) != 2 or any(len(row) != 10 for row in value):
            raise ValueError('Each matrix must be of shape 2x10')
        if not all(0 <= num <= 1 for row in value for num in row):
            raise ValueError('All values must be between 0 and 1')
        return value

class PredictionInput(BaseModel):
    matrix1: Matrix
    matrix2: Matrix

    class Config:
        schema_extra = {
            "example": {
                "matrix1": {
                    "data": [
                        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
                    ]
                },
                "matrix2": {
                    "data": [
                        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4],
                        [0.4, 0.3, 0.2, 0.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
                    ]
                }
            }
        }
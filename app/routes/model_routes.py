# app/routes/model_routes.py

from fastapi import APIRouter, HTTPException
from ..schemas import PredictionInput
import pandas as pd
import torch
import os

router = APIRouter()

# Load the trained PyTorch model (adjust the path as necessary)
model_path = "app/models/dqn_truck_fleet.pth"
if not os.path.exists(model_path):
    raise HTTPException(status_code=500, detail="Model not found")

model = torch.load(model_path)
model.eval()  # Set the model to evaluation mode

@router.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Combine the two matrices into one DataFrame
        matrix1_df = pd.DataFrame(input_data.matrix1.data)
        matrix2_df = pd.DataFrame(input_data.matrix2.data)
        combined_df = pd.concat([matrix1_df, matrix2_df], axis=0).reset_index(drop=True)

        # Convert the DataFrame to a PyTorch tensor
        input_tensor = torch.tensor(combined_df.values, dtype=torch.float32)

        # Perform the prediction
        with torch.no_grad():
            prediction = model(input_tensor).numpy()
        
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

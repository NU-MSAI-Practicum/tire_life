from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import numpy as np
import os
import torch
from .env import TruckFleetEnv
from .dqn import DQNAgent

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="app/templates")

# Define the request and response schemas
class InitialState(BaseModel):
    initial_state: List[List[float]]

class PredictionResponse(BaseModel):
    State: List[float]
    Action: List[int]
    NextState: List[float]
    Reward: float
    Log: str

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=List[PredictionResponse])
def predict_tire_management(data: InitialState):
    num_trucks = len(data.initial_state)
    num_tires_per_truck = len(data.initial_state[0])
    health_threshold = 0.09
    max_steps = 20
    num_steps = 20

    initial_state = np.array(data.initial_state).tolist()
    logs_folder = "./app/logs"
    model_path = "./app/models/dqn_truck_fleet.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = TruckFleetEnv(num_trucks, num_tires_per_truck, health_threshold, max_steps)
    env.reset(initial_state=initial_state)
    
    state_dim = env.num_trucks * env.num_tires_per_truck
    action_dims = {
        "replace": [num_trucks, num_tires_per_truck],
        "swap": [num_trucks, num_trucks, 2, num_tires_per_truck - 2]
    }

    agent = DQNAgent(state_dim, action_dims, device=device)
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
    agent.policy_net.eval()

    predictions = agent.generate_predictions(env, np.array(initial_state), num_steps)

    return [
        PredictionResponse(
            State=pred["State"].tolist(),
            Action=pred["Action"].tolist(),
            NextState=pred["Next State"].tolist(),
            Reward=pred["Reward"],
            Log=pred["Log"]
        )
        for pred in predictions
    ]

@app.post("/train")
def train_model():
    logs_folder = "./app/logs"
    metrics_folder = "./app/metrics"
    model_folder = "./app/models"

    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)
    if not os.path.exists(metrics_folder):
        os.makedirs(metrics_folder)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    from .train import train
    train(logs_folder, metrics_folder, model_folder)
    return {"message": "Training completed and model saved."}

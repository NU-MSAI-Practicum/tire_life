from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import numpy as np
import torch

from .env import TruckFleetEnv
from .dqn import DQNAgent

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

class PredictionRequest(BaseModel):
    num_trucks: int
    num_tires_per_truck: int
    health_threshold: float
    max_steps: int
    num_steps: int
    initial_state: List[List[float]]

@app.get("/", response_class=HTMLResponse)
async def read_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "Home"})

@app.get("/about", response_class=HTMLResponse)
async def read_about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request, "title": "About Us"})

@app.get("/contact", response_class=HTMLResponse)
async def read_contact(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request, "title": "Contact Us"})

@app.post("/submit_contact", response_class=HTMLResponse)
async def submit_contact(request: Request, name: str = Form(...), email: str = Form(...), message: str = Form(...)):
    return templates.TemplateResponse("contact.html", {"request": request, "title": "Contact Us", "message": "Thank you for your message!"})

@app.get("/faqs", response_class=HTMLResponse)
async def read_faqs(request: Request):
    return templates.TemplateResponse("faqs.html", {"request": request, "title": "FAQs"})

@app.post("/predict", response_class=HTMLResponse)
async def predict_actions(
    request: Request,
    num_trucks: int = Form(...),
    num_tires: int = Form(...),
    **tire_data: float
):
    try:
        health_threshold = 0.09
        max_steps = 20
        num_steps = 20
        
        initial_state = []
        for truck in range(1, num_trucks + 1):
            truck_tires = []
            for tire in range(1, num_tires + 1):
                tire_value = float(tire_data.get(f'truck{truck}_tire{tire}', 0.0))
                truck_tires.append(tire_value)
            initial_state.append(truck_tires)

        initial_state = np.array(initial_state)

        # Load the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "./app/models/dqn_truck_fleet_v1.pth"
        
        env = TruckFleetEnv(num_trucks, num_tires, health_threshold, max_steps)
        env.reset(initial_state=initial_state)
        
        state_dim = env.num_trucks * env.num_tires_per_truck
        action_dims = {
            "replace": [num_trucks, num_tires],
            "swap": [num_trucks, num_trucks, 2, num_tires - 2]
        }

        agent = DQNAgent(state_dim, action_dims, device=device)
        agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
        agent.policy_net.eval()

        predictions = agent.generate_predictions(env, initial_state, num_steps)
        detailed_log = agent.save_detailed_log_to_dict(initial_state, predictions)

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "title": "Prediction Results",
                "predictions": predictions,
                "detailed_log": detailed_log
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "title": "Home",
                "error": str(e)
            }
        )

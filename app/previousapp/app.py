from fastapi import FastAPI, HTTPException, Query, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import List
import numpy as np
import pandas as pd
from predict import Predictor

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Load the model
predictor = Predictor(model_path='app/models/dqn_truck_fleet.pth')

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict_from_form(request: Request, num_trucks: int = Form(...), tires: List[str] = Form(...)):
    try:
        # Parse the state from the form input
        tire_values = [list(map(float, tire.split(','))) for tire in tires]
        state_array = np.array(tire_values)
        
        if state_array.shape != (num_trucks, predictor.env.num_tires_per_truck):
            raise ValueError("State size does not match the expected number of trucks and tires")
        
        initial_state, final_state, log_file_path = predictor.predict(state_array)

        # Read the log file
        log_df = pd.read_excel(log_file_path)

        # Convert the action logs to a list of dictionaries for rendering in the template
        log_data = []
        for i in range(num_trucks):
            truck_actions = log_df.loc[log_df['Truck'] == f'Truck {i}', 'Actions'].values[0]
            log_data.append({
                "Truck": f'Truck {i}',
                "Actions": eval(truck_actions)  # Convert string representation of list to actual list
            })

        return templates.TemplateResponse("result.html", {
            "request": request,
            "initial_state": initial_state.tolist(),
            "final_state": final_state.tolist(),
            "log_data": log_data
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

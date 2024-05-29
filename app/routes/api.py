# app/routes/api.py
from fastapi import APIRouter
from ..models import PredictionInput
from ..predict import predict_and_save

router = APIRouter()

@router.post("/api/predict")
async def api_predict(data: PredictionInput):
    output_file = "app/static/results/predictions.xlsx"
    detailed_log_file = "app/static/results/detailed_log.xlsx"
    predict_and_save("app/static/results", "app/models/dqn_truck_fleet_v1.pth", data.num_trucks, data.num_tires_per_truck, 0.09, 20, 20, data.initial_state, output_file, detailed_log_file)
    return {"initial_state": data.initial_state, "output_file": output_file, "detailed_log_file": detailed_log_file}

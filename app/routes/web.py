# app/routes/web.py
from fastapi import APIRouter, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from typing import List
from app.predict import predict_and_save
from io import BytesIO

router = APIRouter()

templates = Jinja2Templates(directory="app/templates")

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@router.get("/contact", response_class=HTMLResponse)
async def contact(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})

@router.get("/faq", response_class=HTMLResponse)
async def faq(request: Request):
    return templates.TemplateResponse("faq.html", {"request": request})

@router.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, num_trucks: int = Form(...), num_tires_per_truck: int = Form(...), file: UploadFile = File(...)):
    contents = await file.read()  # Read the file contents into memory
    df = pd.read_excel(BytesIO(contents))  # Use BytesIO to read the file content
    initial_state = df.to_numpy().tolist()
    output_file = "app/static/results/predictions.xlsx"
    detailed_log_file = "app/static/results/detailed_log.xlsx"
    predict_and_save("app/static/results", "app/models/dqn_truck_fleet_v1.pth", num_trucks, num_tires_per_truck, 0.09, 20, 20, initial_state, output_file, detailed_log_file)
    
    return templates.TemplateResponse("result.html", {"request": request, "initial_state": initial_state, "output_file": output_file, "detailed_log_file": detailed_log_file})

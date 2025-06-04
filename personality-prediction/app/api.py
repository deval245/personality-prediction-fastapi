from fastapi import FastAPI
from app.routes import router 
from app.schemas import PredictionRequest
 # Assuming you saved your prediction route in routes.py

app = FastAPI()

app.include_router(router, prefix="/model")

from fastapi import APIRouter
from app.schemas import Features,PredictionRequest
from app.model_loader import load_model
import torch
import numpy as np

router = APIRouter()
model = load_model()

@router.post("/predict")
async def predict(payload: PredictionRequest):
    features = payload.input_features

    input_data = [[
        features.Time_spent_Alone,
        features.Social_event_attendance,
        features.Going_outside,
        features.Friends_circle_size,
        features.Post_frequency,
        features.Stage_fear_Yes,
        features.Drained_after_socializing_Yes
    ]]

    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor).item()
        predicted_label = int(output >= 0.5)

    personality = "Extrovert" if predicted_label else "Introvert"
    explanation = (
        "You are likely more extroverted, based on your social activity and comfort in groups."
        if predicted_label else
        "You are likely more introverted, based on your comfort with solitude and low social activity."
    )

    return {
        "input_features": payload.input_features.dict(),
        "predicted_personality": personality,
        "confidence_score": round(output, 4),
        "explanation": explanation
    }

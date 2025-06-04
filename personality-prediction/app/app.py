from fastapi import APIRouter
from app.schemas import Features
from app.model_loader import load_model
import torch
import numpy as np

router = APIRouter()
model = load_model()

@router.post("/predict")
async def predict(features: Features):
    input_data = np.array([[  # Keep feature order same
        features.Time_spent_Alone,
        features.Social_event_attendance,
        features.Going_outside,
        features.Friends_circle_size,
        features.Post_frequency,
        features.Stage_fear_Yes,
        features.Drained_after_socializing_Yes
    ]], dtype=np.float32)

    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor).item()
        predicted_label = int(output >= 0.5)

    return {"prediction": predicted_label, "probability": round(output, 4)}

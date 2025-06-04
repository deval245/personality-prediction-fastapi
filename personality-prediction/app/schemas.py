from pydantic import BaseModel, Field
from enum import Enum

class YesNoEnum(int, Enum):
    NO = 0
    YES = 1

class Features(BaseModel):
    Time_spent_Alone: int = Field(..., ge=0, le=24, description="Hours spent alone per day (0–24)")
    Social_event_attendance: int = Field(..., ge=0, le=7, description="Social events attended per week (0–7)")
    Going_outside: int = Field(..., ge=0, le=7, description="Times you go outside per week (0–7)")
    Friends_circle_size: int = Field(..., ge=0, le=50, description="Number of close friends (0–50)")
    Post_frequency: int = Field(..., ge=0, le=10, description="Social media posts per week (0–10)")
    Stage_fear_Yes: YesNoEnum = Field(..., description="Do you have stage fear?")
    Drained_after_socializing_Yes: YesNoEnum = Field(..., description="Do you feel drained after socializing?")

class PredictionRequest(BaseModel):
    input_features: Features

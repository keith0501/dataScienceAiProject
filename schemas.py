from typing import List, Optional
from pydantic import BaseModel, ConfigDict

class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    driver_top3_pct_lag1: float
    driver_top3_pct_lag2: float
    driver_top3_pct_lag3: float
    constructor_top3_pct_lag1: float
    constructor_top3_pct_lag2: float
    constructor_top3_pct_lag3: float
    driver_avg_finish_lag1: float
    driver_avg_finish_lag2: float
    driver_avg_finish_lag3: float
    constructor_avg_finish_lag1: float
    constructor_avg_finish_lag2: float
    constructor_avg_finish_lag3: float
    quali_position: float
    quali_consistency_1y: float
    quali_consistency_2y: float
    quali_consistency_3y: float
    driver_weather_avg_finish_last3y: float
    weather_dry: int = 0
    weather_unknown: int = 0
    weather_variable: int = 0
    weather_wet: int = 0

class BatchPredictionInput(BaseModel):
    rows: List[PredictionInput]

class PredictionResponse(BaseModel):
    top3_probability: float
    predicted_top3: int
    threshold: float

class PodiumRequestRow(PredictionInput):
    driver_name: Optional[str] = None
    constructor_name: Optional[str] = None

class RacePodiumRequest(BaseModel):
    race_id: Optional[int] = None
    rows: List[PodiumRequestRow]

from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI, HTTPException

from schemas import PredictionInput, BatchPredictionInput, PredictionResponse, RacePodiumRequest
from utils import load_artifacts, prepare_input

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model, features, threshold, model_info = load_artifacts()
    state["model"] = model
    state["features"] = features
    state["threshold"] = threshold
    state["model_info"] = model_info
    yield

app = FastAPI(
    title="F1 XGBoost Prediction API",
    version="1.0.0",
    description="FastAPI backend for F1 top-3 finish prediction using XGBoost",
    lifespan=lifespan
)

@app.get("/")
def root():
    return {"message": "F1 XGBoost Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    return state["model_info"]

@app.post("/predict", response_model=PredictionResponse)
def predict(row: PredictionInput):
    try:
        df = pd.DataFrame([row.model_dump()])
        X = prepare_input(df, state["features"])
        prob = float(state["model"].predict_proba(X)[0][1])
        pred = int(prob >= state["threshold"])
        return {"top3_probability": prob, "predicted_top3": pred, "threshold": float(state["threshold"])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
def predict_batch(data: BatchPredictionInput):
    try:
        rows = [r.model_dump() for r in data.rows]
        df = pd.DataFrame(rows)
        X = prepare_input(df, state["features"])
        probs = state["model"].predict_proba(X)[:, 1]
        preds = (probs >= state["threshold"]).astype(int)
        return {
            "threshold": float(state["threshold"]),
            "predictions": [
                {"row_index": i, "top3_probability": float(probs[i]), "predicted_top3": int(preds[i])}
                for i in range(len(df))
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-race-podium")
def predict_race_podium(data: RacePodiumRequest):
    try:
        rows = [r.model_dump() for r in data.rows]
        df = pd.DataFrame(rows)
        X = prepare_input(df, state["features"])
        probs = state["model"].predict_proba(X)[:, 1]
        df["top3_probability"] = probs
        podium = (
            df.sort_values("top3_probability", ascending=False)
            .head(3)
            [["driver_name", "constructor_name", "top3_probability"]]
            .to_dict(orient="records")
        )
        return {
            "race_id": data.race_id,
            "threshold": float(state["threshold"]),
            "predicted_podium": [
                {
                    "driver_name": row.get("driver_name"),
                    "constructor_name": row.get("constructor_name"),
                    "top3_probability": float(row["top3_probability"])
                }
                for row in podium
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import json
import joblib
import pandas as pd

from config import ARTIFACT_DIR


def load_artifacts():
    model = joblib.load(ARTIFACT_DIR / "xgboost_model.pkl")
    with open(ARTIFACT_DIR / "xgboost_feature_columns.json", "r") as f:
        features = json.load(f)
    with open(ARTIFACT_DIR / "xgboost_threshold.json", "r") as f:
        threshold = json.load(f)["best_threshold"]
    with open(ARTIFACT_DIR / "xgboost_model_info.json", "r") as f:
        model_info = json.load(f)
    return model, features, threshold, model_info


def prepare_input(df: pd.DataFrame, features: list[str]):
    df = df.copy()
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df[features]
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

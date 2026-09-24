# ============================================================
# Этап 4. FastAPI backend
# Запуск:  uvicorn api:app --reload --port 8000
# Требует файл model_pipeline.joblib (создаётся в этапе 3) в той же папке.
# ============================================================

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = "model_pipeline.joblib"
DATA_PATH = "real_estate_data.csv"

app = FastAPI(title="Real Estate Price API", version="1.0")

# ---------- Загрузка модели один раз при старте приложения ----------
artifact = joblib.load(MODEL_PATH)
pipeline = artifact["pipeline"]
feature_columns = artifact["feature_columns"]
category_values = artifact["category_values"]
mae_test = artifact["mae_test"]
best_model_name = artifact["best_model_name"]
metrics_validation = artifact["metrics_validation"]
metrics_test_best_model = artifact["metrics_test_best_model"]


# ---------- Схемы запроса/ответа ----------
class PredictRequest(BaseModel):
    sub_type: str = Field(..., description="Тип недвижимости, например 'Daire'")
    heating_type: str = Field(..., description="Тип отопления, например 'Yok'")
    size: float = Field(..., gt=0, le=5000, description="Площадь, м²")
    room_count: float = Field(..., ge=0, le=20, description="Количество комнат (сумма, напр. 3+1 -> 4)")
    building_age: float = Field(..., ge=0, le=100, description="Возраст здания, лет")
    total_floor_count: float = Field(..., ge=1, le=100, description="Этажность дома")
    tom: float = Field(..., ge=0, le=3650, description="Срок размещения объявления, дней")
    listing_days: float = Field(30, ge=0, le=3650, description="Длительность объявления, дней")


class PredictResponse(BaseModel):
    predicted_price: float
    price_low: float
    price_high: float
    model_name: str
    mae: float


class ModelInfoResponse(BaseModel):
    best_model_name: str
    category_values: dict
    metrics_validation: list
    metrics_test_best_model: dict


# ---------- Эндпоинты ----------
@app.get("/health")
def health():
    return {"status": "ok", "model": best_model_name}


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    return ModelInfoResponse(
        best_model_name=best_model_name,
        category_values=category_values,
        metrics_validation=metrics_validation,
        metrics_test_best_model=metrics_test_best_model,
    )


@app.get("/data/summary")
def data_summary():
    """Базовая сводка по датасету для дашборда."""
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[(df["price_currency"] == "TRY") & df["price"].notna() & (df["price"] > 0)]
    by_subtype = (
        df.groupby("sub_type")["price"].mean().sort_values(ascending=False).round(0).to_dict()
    )
    return {
        "rows": int(len(df)),
        "avg_price": float(df["price"].mean()),
        "median_price": float(df["price"].median()),
        "avg_price_by_subtype": by_subtype,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.sub_type not in category_values["sub_type"]:
        raise HTTPException(400, f"Неизвестный sub_type: {req.sub_type}")
    if req.heating_type not in category_values["heating_type"]:
        raise HTTPException(400, f"Неизвестный heating_type: {req.heating_type}")

    row = {
        "sub_type": req.sub_type,
        "heating_type": req.heating_type,
        "size": req.size,
        "size_log": float(np.log1p(req.size)),
        "room_count_num": req.room_count,
        "building_age_num": req.building_age,
        "total_floor_num": req.total_floor_count,
        "tom": req.tom,
        "listing_days": req.listing_days,
    }
    X = pd.DataFrame([row])[feature_columns]

    try:
        pred = float(pipeline.predict(X)[0])
    except Exception as exc:
        raise HTTPException(500, f"Ошибка предсказания: {exc}")

    return PredictResponse(
        predicted_price=round(pred, 2),
        price_low=round(max(pred - mae_test, 0), 2),
        price_high=round(pred + mae_test, 2),
        model_name=best_model_name,
        mae=round(mae_test, 2),
    )

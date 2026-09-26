"""
AI Service — FastAPI
--------------------------------
Tự động load model.joblib ngay khi container khởi động (startup event).
Endpoints:
  GET  /health   -> kiểm tra service + model đã sẵn sàng chưa
  POST /predict  -> dự đoán xác suất nguy cơ đột quỵ
  GET  /schema   -> trả schema.json cho Frontend tự sinh form
"""

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai-service")

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(MODELS_DIR, "model.joblib"))
SCHEMA_PATH = os.environ.get("SCHEMA_PATH", os.path.join(MODELS_DIR, "schema.json"))
METADATA_PATH = os.environ.get("METADATA_PATH", os.path.join(MODELS_DIR, "metadata.json"))

state = {"model": None, "schema": None, "metadata": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Đang load model từ {MODEL_PATH} ...")
    if not os.path.exists(MODEL_PATH):
        logger.warning("KHÔNG tìm thấy model.joblib — /predict sẽ trả lỗi 503 cho tới khi có model.")
    else:
        state["model"] = joblib.load(MODEL_PATH)
        logger.info("Model đã sẵn sàng.")
    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH, encoding="utf-8") as f:
            state["schema"] = json.load(f)
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, encoding="utf-8") as f:
            state["metadata"] = json.load(f)
    yield
    state.clear()


app = FastAPI(
    title="Stroke Risk AI Service",
    description="Dự đoán nguy cơ đột quỵ dựa trên chỉ số sink học và thói quen sinh hoạt.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    gender: Literal["Male", "Female", "Other"]
    age: float = Field(..., ge=0, le=120)
    hypertension: Literal[0, 1]
    heart_disease: Literal[0, 1]
    ever_married: Literal["Yes", "No"]
    work_type: Literal["children", "Govt_job", "Never_worked", "Private", "Self-employed"]
    Residence_type: Literal["Rural", "Urban"]
    avg_glucose_level: float = Field(..., ge=0)
    bmi: Optional[float] = Field(None, ge=0)
    smoking_status: Literal["formerly smoked", "never smoked", "smokes", "Unknown"]


class PredictResponse(BaseModel):
    request_id: str
    stroke_risk_probability: float
    stroke_prediction: int
    risk_level: str
    model_name: str
    inference_ms: float


def _risk_level(p: float) -> str:
    if p < 0.2:
        return "Thấp"
    if p < 0.5:
        return "Trung bình"
    return "Cao"


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    response = await call_next(request)
    dt = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(f"[{request_id}] {request.method} {request.url.path} -> {response.status_code} ({dt:.1f} ms)")
    return response


@app.get("/health")
def health():
    return {
        "status": "ok" if state["model"] is not None else "model_not_loaded",
        "model_loaded": state["model"] is not None,
        "model_name": (state["metadata"] or {}).get("model_name"),
    }


@app.get("/schema")
def get_schema():
    if state["schema"] is None:
        raise HTTPException(status_code=404, detail="schema.json chưa được nạp.")
    return state["schema"]


@app.get("/metadata")
def get_metadata():
    if state["metadata"] is None:
        raise HTTPException(status_code=404, detail="metadata.json chưa được nạp.")
    return state["metadata"]


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng.")

    request_id = str(uuid.uuid4())
    row = pd.DataFrame([payload.model_dump()])

    t0 = time.perf_counter()
    try:
        proba = float(state["model"].predict_proba(row)[0, 1])
    except Exception as exc:
        logger.exception(f"[{request_id}] Lỗi khi dự đoán")
        raise HTTPException(status_code=422, detail=f"Lỗi dữ liệu đầu vào: {exc}") from exc
    inference_ms = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        request_id=request_id,
        stroke_risk_probability=round(proba, 4),
        stroke_prediction=int(proba >= 0.5),
        risk_level=_risk_level(proba),
        model_name=(state["metadata"] or {}).get("model_name", "unknown"),
        inference_ms=round(inference_ms, 3),
    )

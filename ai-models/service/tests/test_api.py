import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from main import app

# Dùng context manager để kích hoạt lifespan (load model) trước khi test
client = TestClient(app)
client.__enter__()

SAMPLE_PAYLOAD = {
    "gender": "Male",
    "age": 67,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked",
}


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "model_loaded" in resp.json()


def test_schema():
    resp = client.get("/schema")
    assert resp.status_code == 200
    assert resp.json()["target"] == "stroke"


def test_predict_success():
    resp = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["stroke_risk_probability"] <= 1.0
    assert body["stroke_prediction"] in (0, 1)
    assert body["risk_level"] in ("Thấp", "Trung bình", "Cao")


def test_predict_missing_bmi_ok():
    payload = dict(SAMPLE_PAYLOAD)
    payload["bmi"] = None
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200


def test_predict_invalid_field():
    payload = dict(SAMPLE_PAYLOAD)
    payload["gender"] = "invalid"
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422

"""
train.py
--------------------------------
Huấn luyện 4 model trên cùng 1 pipeline tiền xử lý, cùng 1 tập
Stratified Train/Test = 80/20, ưu tiên tối ưu Recall (lớp 1) vì bài toán
y tế cần giảm thiểu tối đa Âm tính giả (False Negative).

Chạy: python ai-models/src/train.py
Kết quả: ai-models/models/comparison.json (so sánh 4 model)
         ai-models/models/candidates/*.joblib (pipeline đầy đủ của từng model)
"""

import os
import json
import time
import warnings

import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, roc_auc_score, precision_score, f1_score

from preprocess import load_raw_data, split_features_target, build_preprocessor

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CANDIDATES_DIR = os.path.join(MODELS_DIR, "candidates")
os.makedirs(CANDIDATES_DIR, exist_ok=True)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

MODEL_GRID = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "params": {"clf__C": [0.01, 0.1, 1, 10]},
    },
    "naive_bayes": {
        "estimator": GaussianNB(),
        "params": {"clf__var_smoothing": [1e-9, 1e-8, 1e-7]},
    },
    "svm_rbf": {
        "estimator": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        "params": {"clf__C": [1, 10], "clf__gamma": ["scale", "auto"]},
    },
    "random_forest": {
        "estimator": RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        "params": {"clf__n_estimators": [200], "clf__max_depth": [6, 12, None]},
    },
}


def get_train_test_split():
    df = load_raw_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def train_one(name: str, spec: dict, X_train, y_train, X_test, y_test) -> dict:
    preprocessor = build_preprocessor()
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("clf", spec["estimator"])])

    grid = GridSearchCV(
        pipe, param_grid=spec["params"], scoring="recall", cv=CV, n_jobs=-1, refit=True
    )
    grid.fit(X_train, y_train)
    best_pipe = grid.best_estimator_

    t0 = time.perf_counter()
    y_pred = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1]
    inference_ms = (time.perf_counter() - t0) / len(X_test) * 1000

    model_path = os.path.join(CANDIDATES_DIR, f"{name}.joblib")
    joblib.dump(best_pipe, model_path, compress=3)
    size_kb = os.path.getsize(model_path) / 1024

    metrics = {
        "model": name,
        "best_params": grid.best_params_,
        "recall_class1": round(recall_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "precision_class1": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "f1_class1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "inference_ms_per_sample": round(inference_ms, 3),
        "file_size_kb": round(size_kb, 1),
        "model_path": model_path,
    }
    print(f"[{name}] Recall={metrics['recall_class1']} ROC-AUC={metrics['roc_auc']} "
          f"Precision={metrics['precision_class1']} F1={metrics['f1_class1']} "
          f"({metrics['inference_ms_per_sample']} ms/sample, {metrics['file_size_kb']} KB)")
    return metrics


def main():
    X_train, X_test, y_train, y_test = get_train_test_split()
    print(f"Train: {len(X_train)} | Test: {len(X_test)} | Stroke=1 trong test: {y_test.mean():.3%}\n")

    results = []
    for name, spec in MODEL_GRID.items():
        results.append(train_one(name, spec, X_train, y_train, X_test, y_test))

    comparison_path = os.path.join(MODELS_DIR, "comparison.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu bảng so sánh 4 model tại: {comparison_path}")


if __name__ == "__main__":
    main()

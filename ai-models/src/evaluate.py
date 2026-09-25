"""
evaluate.py
--------------------------------
Đọc ai-models/models/comparison.json (do train.py sinh ra), chọn model tốt
nhất theo tiêu chí ưu tiên Recall (lớp 1) rồi ROC-AUC, sau đó đóng gói:
- ai-models/models/model.joblib   (Sklearn Pipeline: Preprocess + Model)
- ai-models/models/schema.json    (Frontend dùng để tự sinh form nhập liệu)
- ai-models/models/metadata.json  (Thông tin phiên bản, metric, ngày huấn luyện)

Chạy: python ai-models/src/evaluate.py
"""

import os
import json
import shutil
from datetime import datetime, timezone

import joblib
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import (
    load_raw_data, split_features_target,
    NUMERIC_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES,
)
from train import get_train_test_split

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

COMPARISON_PATH = os.path.join(MODELS_DIR, "comparison.json")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")
SCHEMA_PATH = os.path.join(MODELS_DIR, "schema.json")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

FEATURE_META = {
    "age": {"type": "number", "label": "Tuổi", "min": 0, "max": 120},
    "avg_glucose_level": {"type": "number", "label": "Mức đường huyết trung bình", "min": 40, "max": 300},
    "bmi": {"type": "number", "label": "Chỉ số khối cơ thể (BMI)", "min": 10, "max": 100},
    "hypertension": {"type": "boolean", "label": "Tiền sử cao huyết áp", "values": [0, 1]},
    "heart_disease": {"type": "boolean", "label": "Tiền sử bệnh tim", "values": [0, 1]},
    "gender": {"type": "category", "label": "Giới tính", "values": ["Male", "Female", "Other"]},
    "ever_married": {"type": "category", "label": "Đã từng kết hôn", "values": ["Yes", "No"]},
    "work_type": {"type": "category", "label": "Loại hình công việc",
                  "values": ["children", "Govt_job", "Never_worked", "Private", "Self-employed"]},
    "Residence_type": {"type": "category", "label": "Nơi sinh sống", "values": ["Rural", "Urban"]},
    "smoking_status": {"type": "category", "label": "Tình trạng hút thuốc",
                        "values": ["formerly smoked", "never smoked", "smokes", "Unknown"]},
}


MIN_PRECISION = 0.15  # loại các model suy biến kiểu "đoán gần như tất cả là dương tính"


def pick_best_model(results: list) -> dict:
    """
    Ưu tiên hàng đầu: Recall (lớp 1) và ROC-AUC, nhưng loại các model suy biến
    (Recall rất cao chỉ vì gần như luôn đoán 1, khiến Precision cực thấp — không
    có giá trị sử dụng thực tế). Trong các model còn lại (đủ cân bằng, không
    overfit), chọn Recall cao nhất, hoà thì xét ROC-AUC.
    """
    candidates = [r for r in results if r.get("model_type", "candidate") != "baseline"]
    if not candidates:
        raise ValueError("comparison.json không có model candidate để triển khai.")
    balanced = [r for r in candidates if r["precision_class1"] >= MIN_PRECISION]
    pool = balanced if balanced else candidates
    return sorted(pool, key=lambda r: (r["recall_class1"], r["roc_auc"]), reverse=True)[0]


def make_eda_figures():
    df = load_raw_data()

    sns.set_theme(style="whitegrid")

    # 1. Phân phối target
    plt.figure(figsize=(5, 4))
    sns.countplot(x="stroke", data=df, palette=["#4C72B0", "#C44E52"])
    plt.title("Phân phối nhãn Stroke (mất cân bằng)")
    plt.xlabel("stroke"); plt.ylabel("Số lượng")
    plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, "01_target_distribution.png"), dpi=120); plt.close()

    # 2. Phân phối tuổi theo stroke
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="age", hue="stroke", bins=30, kde=True, palette=["#4C72B0", "#C44E52"])
    plt.title("Phân phối tuổi theo nhãn Stroke")
    plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, "02_age_distribution.png"), dpi=120); plt.close()

    # 3. Glucose vs stroke
    plt.figure(figsize=(5, 4))
    sns.boxplot(x="stroke", y="avg_glucose_level", data=df, palette=["#4C72B0", "#C44E52"])
    plt.title("Mức đường huyết trung bình theo nhãn Stroke")
    plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, "03_glucose_boxplot.png"), dpi=120); plt.close()

    # 4. Missing values (bmi)
    plt.figure(figsize=(5, 4))
    missing = df.isna().sum()
    missing = missing[missing > 0]
    sns.barplot(x=missing.index, y=missing.values, color="#DD8452")
    plt.title("Số lượng giá trị khuyết theo cột")
    plt.ylabel("Số dòng missing")
    plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, "04_missing_values.png"), dpi=120); plt.close()

    # 5. Ma trận tương quan các biến số
    plt.figure(figsize=(5, 4))
    numeric_df = df[["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease", "stroke"]]
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Ma trận tương quan")
    plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, "05_correlation_heatmap.png"), dpi=120); plt.close()

    print(f"Đã lưu 5 hình EDA vào {FIGURES_DIR}")


def make_confusion_matrix_figure(best_name: str):
    X_train, X_test, y_train, y_test = get_train_test_split()
    candidate_path = os.path.join(MODELS_DIR, "candidates", f"{best_name}.joblib")
    pipe = joblib.load(candidate_path)
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "06_confusion_matrix_best_model.png"), dpi=120)
    plt.close()


def build_schema() -> dict:
    return {
        "target": "stroke",
        "features": [
            {"name": f, **FEATURE_META[f]}
            for f in NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
        ],
    }


def main():
    if not os.path.exists(COMPARISON_PATH):
        raise FileNotFoundError("Chưa có comparison.json. Hãy chạy train.py trước.")

    with open(COMPARISON_PATH, encoding="utf-8") as f:
        results = json.load(f)

    best = pick_best_model(results)
    best_name = best["model"]
    print(f"Model được chọn: {best_name} (Recall={best['recall_class1']}, ROC-AUC={best['roc_auc']})")

    # 1) model.joblib — copy pipeline tốt nhất
    shutil.copyfile(os.path.join(MODELS_DIR, "candidates", f"{best_name}.joblib"), FINAL_MODEL_PATH)

    # 2) schema.json
    schema = build_schema()
    with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    # 3) metadata.json
    metadata = {
        "model_name": best_name,
        "selected_reason": "Đã loại model có Precision lớp 1 dưới 0.15 nếu còn model hợp lệ; sau đó chọn Recall test cao nhất, hòa thì chọn ROC-AUC cao hơn.",
        "selection_rule": {
            "minimum_precision_class1": MIN_PRECISION,
            "primary_metric": "recall_class1",
            "tie_breaker": "roc_auc",
        },
        "metrics": {k: v for k, v in best.items() if k not in ("model", "model_path", "best_params")},
        "best_params": best["best_params"],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_models_compared": [r["model"] for r in results],
        "sklearn_pipeline_steps": ["preprocess (ColumnTransformer)", "clf"],
        "target": "stroke",
        "positive_class": 1,
        "note": "Model được huấn luyện trên dữ liệu giả lập (synthetic) bám schema Kaggle. "
                "Thay dataset thật vào ai-models/data/ rồi chạy lại train.py + evaluate.py trước khi dùng thật.",
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    make_eda_figures()
    make_confusion_matrix_figure(best_name)

    print(f"\nĐã đóng gói xong:\n- {FINAL_MODEL_PATH}\n- {SCHEMA_PATH}\n- {METADATA_PATH}")


if __name__ == "__main__":
    main()

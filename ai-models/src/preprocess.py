"""
preprocess.py
--------------------------------
Xây dựng ColumnTransformer dùng chung cho train/evaluate/service, đảm bảo
pipeline lúc suy luận (inference) xử lý dữ liệu giống hệt lúc huấn luyện.
"""

import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "healthcare-dataset-stroke-data.csv")

NUMERIC_FEATURES = ["age", "avg_glucose_level", "bmi"]
BINARY_FEATURES = ["hypertension", "heart_disease"]
CATEGORICAL_FEATURES = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
TARGET = "stroke"
DROP_COLS = ["id"]

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Hãy giải nén ai-models/data/dataset.zip trước, "
            f"hoặc chạy ai-models/src/make_synthetic_dataset.py để tạo dữ liệu mẫu."
        )
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    # Loại một số outlier hiếm gặp trong bản gốc (gender = "Other" chỉ có 1 dòng)
    return df


def split_features_target(df: pd.DataFrame):
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    binary_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("bin", binary_pipeline, BINARY_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


if __name__ == "__main__":
    df = load_raw_data()
    X, y = split_features_target(df)
    print(f"Số dòng: {len(df)} | Số đặc trưng: {X.shape[1]}")
    print(f"Tỷ lệ stroke=1: {y.mean():.3%}")
    print(f"Missing values:\n{X.isna().sum()[X.isna().sum() > 0]}")

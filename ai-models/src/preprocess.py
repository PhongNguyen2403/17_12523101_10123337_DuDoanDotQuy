"""
preprocess.py
--------------------------------
Xây dựng ColumnTransformer dùng chung cho train/evaluate/service, đảm bảo
pipeline lúc suy luận (inference) xử lý dữ liệu giống hệt lúc huấn luyện.
"""

import os
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "healthcare-dataset-stroke-data.csv")
DATA_ZIP_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.zip")

NUMERIC_FEATURES = ["age", "avg_glucose_level", "bmi"]
BINARY_FEATURES = ["hypertension", "heart_disease"]
CATEGORICAL_FEATURES = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
TARGET = "stroke"
DROP_COLS = ["id"]

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
    elif os.path.exists(DATA_ZIP_PATH):
        with ZipFile(DATA_ZIP_PATH) as archive:
            csv_names = [name for name in archive.namelist() if Path(name).name == Path(DATA_PATH).name]
            if not csv_names:
                raise FileNotFoundError(f"Không tìm thấy {Path(DATA_PATH).name} trong {DATA_ZIP_PATH}.")
            with archive.open(csv_names[0]) as csv_file:
                df = pd.read_csv(csv_file)
    else:
        raise FileNotFoundError(
            f"Không tìm thấy {path} hoặc {DATA_ZIP_PATH}. "
            f"Hãy đặt dataset.zip vào ai-models/data/."
        )
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

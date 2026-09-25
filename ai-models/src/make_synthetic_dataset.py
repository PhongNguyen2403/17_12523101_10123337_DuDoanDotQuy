"""
make_synthetic_dataset.py
--------------------------------
Sinh dữ liệu GIẢ LẬP bám sát đúng schema và phân phối thống kê của bộ dữ liệu
Kaggle "Stroke Prediction Dataset" (fedesoriano):
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

CHỈ DÙNG KHI CHƯA CÓ FILE THẬT. Khi có dataset.zip thật, hãy giải nén đè lên
ai-models/data/healthcare-dataset-stroke-data.csv rồi chạy lại toàn bộ pipeline
(preprocess.py -> train.py -> evaluate.py); KHÔNG cần sửa code.

Đặc điểm được mô phỏng lại cho giống bản gốc:
- 5110 dòng, 12 cột đúng tên/kiểu dữ liệu như mô tả trong README.
- stroke=1 chỉ chiếm ~5% (mất cân bằng nghiêm trọng).
- bmi có khoảng 4% giá trị khuyết (NaN) giống bản gốc.
- Người có stroke=1 thiên về: tuổi cao hơn, hypertension/heart_disease cao hơn,
  avg_glucose_level cao hơn -> để 4 model học được tín hiệu thật sự thay vì nhiễu.
"""

import numpy as np
import pandas as pd
import zipfile
import os

RNG = np.random.default_rng(42)
N = 5110
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, "healthcare-dataset-stroke-data.csv")
ZIP_PATH = os.path.join(OUT_DIR, "dataset.zip")


def sample_age():
    # Phân phối lệch giống dân số thực (nhiều trẻ em + người lớn, ít người rất già)
    return np.clip(RNG.gamma(shape=3.0, scale=15.0, size=N), 0.08, 82).round(2)


def main():
    age = sample_age()
    gender = RNG.choice(["Male", "Female", "Other"], size=N, p=[0.414, 0.585, 0.001])
    ever_married = np.where(age > 18, RNG.choice(["Yes", "No"], size=N, p=[0.75, 0.25]),
                             RNG.choice(["No", "Yes"], size=N, p=[0.98, 0.02]))
    work_type = np.select(
        [age < 16, age < 22],
        [np.full(N, "children"), RNG.choice(["Never_worked", "Private", "Self-employed"], size=N, p=[0.5, 0.4, 0.1])],
        default=RNG.choice(["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
                            size=N, p=[0.57, 0.16, 0.13, 0.13, 0.01])
    )
    residence = RNG.choice(["Urban", "Rural"], size=N, p=[0.508, 0.492])
    smoking = RNG.choice(["never smoked", "formerly smoked", "smokes", "Unknown"],
                          size=N, p=[0.37, 0.17, 0.15, 0.31])

    # Nguy cơ nền tăng dần theo tuổi + yếu tố khác (để tạo tín hiệu thật cho model học)
    base_risk = (age / 100) ** 2.5
    hypertension = RNG.binomial(1, np.clip(base_risk * 1.3 + 0.03, 0.01, 0.6))
    heart_disease = RNG.binomial(1, np.clip(base_risk * 1.1 + 0.015, 0.005, 0.5))

    avg_glucose_level = np.clip(
        RNG.normal(loc=90 + base_risk * 120, scale=35, size=N), 55, 280
    ).round(2)

    bmi = np.clip(RNG.normal(loc=28.5, scale=7.5, size=N), 10, 97).round(1)
    # ~4% missing giống bản gốc
    missing_idx = RNG.choice(N, size=int(N * 0.0393), replace=False)
    bmi[missing_idx] = np.nan

    stroke_logit = (
        -7.6
        + 0.055 * age
        + 1.0 * hypertension
        + 1.15 * heart_disease
        + 0.01 * (avg_glucose_level - 90)
        + 0.02 * (np.nan_to_num(bmi, nan=28.5) - 28.5)
        + np.where(smoking == "smokes", 0.35, 0)
        + np.where(smoking == "formerly smoked", 0.2, 0)
    )
    stroke_prob = 1 / (1 + np.exp(-stroke_logit))
    stroke = RNG.binomial(1, stroke_prob)

    df = pd.DataFrame({
        "id": np.arange(1, N + 1) * 7 + 1000,
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking,
        "stroke": stroke,
    })

    df.to_csv(CSV_PATH, index=False)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(CSV_PATH, arcname="healthcare-dataset-stroke-data.csv")

    print(f"Đã tạo {len(df)} dòng.")
    print(f"Tỷ lệ stroke=1: {df['stroke'].mean():.3%}")
    print(f"Tỷ lệ bmi missing: {df['bmi'].isna().mean():.3%}")
    print(f"CSV: {CSV_PATH}")
    print(f"ZIP: {ZIP_PATH}")


if __name__ == "__main__":
    main()

# Mô tả dữ liệu (DATA.md)

## Nguồn dữ liệu
- **Tên:** Stroke Prediction Dataset
- **Tác giả:** fedesoriano
- **Nguồn:** Kaggle — https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- **Giấy phép:** Open Database License (ODbL) v1.0
- **Số dòng:** 5110 bệnh nhân | **Số cột:** 12

## ⚠️ Lưu ý quan trọng
File `dataset.zip` trong thư mục này hiện là **dữ liệu giả lập (synthetic)**,
được sinh bởi `ai-models/src/make_synthetic_dataset.py` để mô phỏng đúng:
- Schema (đúng 12 cột, đúng kiểu dữ liệu, đúng nhãn giá trị).
- Tỷ lệ mất cân bằng lớp mục tiêu (~5% `stroke = 1`).
- Tỷ lệ giá trị khuyết ở cột `bmi` (~4%).
- Quan hệ tương quan hợp lý giữa tuổi/huyết áp/tim mạch/đường huyết và nguy cơ đột quỵ,
  để 4 model có tín hiệu thật để học thay vì nhiễu ngẫu nhiên.

**Trước khi nộp bài / báo cáo kết quả chính thức**, hãy:
1. Tải file thật từ Kaggle (cần tài khoản Kaggle + `kaggle.json` API token).
2. Giải nén đè vào `ai-models/data/healthcare-dataset-stroke-data.csv`.
3. Chạy lại toàn bộ pipeline: `preprocess.py` → `train.py` → `evaluate.py`.
   Không cần sửa code — schema giống hệt nhau.

## Cách tải dữ liệu thật bằng Kaggle CLI
```bash
pip install kaggle
# đặt kaggle.json vào ~/.kaggle/kaggle.json (KHÔNG commit file này lên git)
kaggle datasets download -d fedesoriano/stroke-prediction-dataset -p ai-models/data --unzip
mv ai-models/data/healthcare-dataset-stroke-data.csv ai-models/data/
```

## Giải nén dataset.zip (bản giả lập hiện tại)
```bash
unzip ai-models/data/dataset.zip -d ai-models/data/
```

## Mô tả cột dữ liệu
| Cột | Kiểu | Mô tả |
|---|---|---|
| id | int | Mã định danh bệnh nhân (loại bỏ khi train) |
| gender | category | Male / Female / Other |
| age | float | Tuổi (0–82) |
| hypertension | int(0/1) | Tiền sử cao huyết áp |
| heart_disease | int(0/1) | Tiền sử bệnh tim |
| ever_married | category | Yes / No |
| work_type | category | children / Govt_job / Never_worked / Private / Self-employed |
| Residence_type | category | Urban / Rural |
| avg_glucose_level | float | Mức đường huyết trung bình |
| bmi | float | Body Mass Index (có missing) |
| smoking_status | category | formerly smoked / never smoked / smokes / Unknown |
| stroke | int(0/1) | **Target**: 1 = có nguy cơ đột quỵ |

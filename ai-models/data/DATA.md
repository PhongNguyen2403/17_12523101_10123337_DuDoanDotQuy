# Mô tả dữ liệu

## Nguồn dữ liệu
- **Tên:** Stroke Prediction Dataset
- **Tác giả:** fedesoriano
- **Nguồn:** Kaggle — https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- **Giấy phép:** Open Database License (ODbL) v1.0
- **Số dòng:** 5110 bệnh nhân | **Số cột:** 12

## Trạng thái dữ liệu

> File `dataset.zip` trong thư mục này chứa **dữ liệu thật** từ bộ dữ liệu trên Kaggle, không phải dữ liệu tổng hợp. Không cần tải lại dữ liệu trước khi chạy hoặc báo cáo kết quả.

Trong thư mục hiện tại, dữ liệu được cung cấp dưới dạng ZIP. Notebook đọc CSV bên trong ZIP; `ai-models/src/preprocess.py` cũng đọc ZIP khi không tìm thấy CSV rời. Nếu bổ sung `healthcare-dataset-stroke-data.csv` vào cùng thư mục, script sẽ ưu tiên CSV rời đó, vì vậy hãy bảo đảm CSV và ZIP cùng nguồn và cùng phiên bản.

## Cách notebook và script đọc dữ liệu

Notebook đọc trực tiếp CSV bên trong `dataset.zip`. Script `preprocess.py` đọc CSV rời nếu có, nếu không sẽ đọc CSV trong ZIP. Không cần tạo thư mục `data_from_zip`:

```text
ai-models/data/dataset.zip
└── healthcare-dataset-stroke-data.csv
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
| stroke | int(0/1) | **Nhãn mục tiêu:** 1 = hồ sơ có ghi nhận đột quỵ, 0 = không ghi nhận; đây không phải dự đoán chẩn đoán nguy cơ |

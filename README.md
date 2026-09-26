# Hệ thống Dự đoán Nguy cơ Đột quỵ (Stroke Risk Prediction)

> **Môn học:** Học máy cơ bản (221180) — Lớp 12523W.2
> **Giảng viên hướng dẫn:** Nguyễn Đức Tuấn Anh
> **Phiên bản tài liệu áp dụng:** 2.0 (23/09/2026)

---

## Cấu trúc Repository

```text
17_12523101_10123337_DuDoanDotQuy/
├── app/
│   ├── frontend/               # React (Vite) + Dockerfile
│   └── backend/                # Node.js/Express API + Dockerfile + tests/
├── ai-models/
│   ├── colab/                  # 01_eda, 02_preprocess, 03_train, 04_evaluate
│   ├── src/                    # preprocess.py, train.py, evaluate.py, make_synthetic_dataset.py
│   ├── data/                   # dataset.zip + DATA.md
│   ├── models/                 # model.joblib, candidates/, comparison.json, schema.json, metadata.json
│   ├── service/                # FastAPI prediction API + Dockerfile + tests/
│   └── requirements.txt
├── docs/
│   ├── slide.pptx               # chưa có — bổ sung trước hạn nộp slide
│   ├── baocao.docx              # chưa có — bổ sung trước hạn nộp báo cáo
│   └── figures/                 # 5 hình EDA + confusion matrix + so sánh model
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

Các notebook trong `ai-models/colab/` chạy độc lập theo thứ tự `01_eda` → `02_preprocess`
→ `03_train` → `04_evaluate`. Notebook đọc trực tiếp CSV bên trong `dataset.zip`, không cần
tạo thư mục `data_from_zip`. Khi chạy trên VS Code, artifact trung gian nằm ở `.colab_artifacts/`;
cell cuối của `03_train` đồng bộ pipeline và các model ứng viên vào `ai-models/models/`.

---

## 1. Thành viên nhóm

Số nhóm: 17
Tên repo chuẩn quy ước: `17_12523101_10123337_DuDoanDotQuy`

| STT | Họ và tên | MSSV | Vai trò & Phân công công việc | % Hoàn thành |
|---|---|---|---|---|
| 1 | Nguyễn Thế Phong | 12523101 | Trưởng nhóm: EDA dữ liệu, Huấn luyện 2 Model(logistic regression, random forest), Xây dựng AI Service (FastAPI) & Dockerize, Xây dựng  Frontend (React). | 100% |
| 2 | Lê Quang Trường | 10123337 | Thành viên:Huấn luyện 2 Model (SVM,naive bayes ), Xây dựng Backend (Node.js/Express), Kết nối MongoDB, Deploy Tunnel/Ngrok & Load Test. | 100% |

## 2. Bài toán (Problem Formulation)

Đột quỵ (Stroke) là một trong những nguyên nhân hàng đầu gây tử vong và tàn tật vĩnh viễn trên thế giới.
Bài toán đặt ra là xây dựng mô hình Học máy có khả năng dự đoán sớm nguy cơ đột quỵ của một cá nhân
dựa trên các chỉ số sinh học và yếu tố thói quen sinh hoạt.

- **Loại bài toán:** Phân loại nhị phân (Binary Classification).
- **Cột mục tiêu (target):** `stroke` (0: Không có nguy cơ, 1: Có nguy cơ đột quỵ).
- **Ý nghĩa thực tế:** Cung cấp công cụ hỗ trợ cho y bác sĩ và người dùng tự kiểm tra sức khỏe.
  Do đặc thù y tế, hệ thống ưu tiên giảm thiểu tối đa Âm tính giả (False Negative - FN).

## 3. Dữ liệu (Dataset Information)

- **Nguồn dữ liệu:** Kaggle — [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (ODbL).
- **Lưu trữ repo:** `ai-models/data/dataset.zip` (kèm `DATA.md`).

> ⚠️ **Dữ liệu hiện tại là dữ liệu giả lập (synthetic)** được sinh bởi
> `ai-models/src/make_synthetic_dataset.py` vì môi trường tạo repo này không truy cập được
> Kaggle trực tiếp. Xem chi tiết cách thay bằng dữ liệu thật tại `ai-models/data/DATA.md`.

Notebook và script đọc trực tiếp `healthcare-dataset-stroke-data.csv` bên trong
`ai-models/data/dataset.zip`; không cần giải nén dataset.

### Mô tả các đặc trưng (Features)
- `id`: Mã định danh bệnh nhân (loại bỏ khi huấn luyện).
- `gender`: Giới tính (Male, Female, Other).
- `age`: Tuổi của bệnh nhân (0–82).
- `hypertension`: Tiền sử cao huyết áp (0/1).
- `heart_disease`: Tiền sử bệnh tim (0/1).
- `ever_married`: Trạng thái kết hôn (No, Yes).
- `work_type`: Loại hình công việc.
- `Residence_type`: Loại nơi sinh sống (Rural, Urban).
- `avg_glucose_level`: Mức đường huyết trung bình.
- `bmi`: Body Mass Index (có missing values).
- `smoking_status`: Tình trạng hút thuốc.

## 4. Kết quả Model (Model Evaluation & Selection)

Cả 4 model được huấn luyện trên cùng pipeline tiền xử lý, cùng Stratified Train/Test = 80/20,
GridSearchCV tối ưu theo `recall`. Do dữ liệu mất cân bằng (~5% nhãn 1), chỉ số ưu tiên hàng đầu
là **Recall (lớp 1)** và **ROC-AUC**.

Kết quả thực tế khi chạy trên **dữ liệu giả lập** trong repo này (chạy `train.py` rồi `evaluate.py`
để tái tạo — số liệu sẽ khác khi dùng dataset Kaggle thật):

| Model | Recall (Class 1) | ROC-AUC | Precision (Class 1) | F1-Score | Suy luận/mẫu | Kích thước |
|---|---|---|---|---|---|---|
| **Logistic Regression** ✅ | 0.80 | 0.88 | 0.18 | 0.30 | ~0.01 ms | 2.3 KB |
| Naive Bayes | 1.00* | 0.88 | 0.06* | 0.11 | ~0.015 ms | 2.6 KB |
| SVM (RBF) | 0.78 | 0.88 | 0.18 | 0.29 | ~0.18 ms | 51 KB |
| Random Forest | 0.69 | 0.88 | 0.24 | 0.35 | ~0.05 ms | 542 KB |

\* Naive Bayes đạt Recall tuyệt đối nhưng Precision quá thấp (gần như luôn đoán 1) —
model suy biến, không có giá trị sử dụng thực tế nên **không được chọn** dù Recall cao nhất
(xem tiêu chí lựa chọn trong `ai-models/src/evaluate.py`).

**Model được chọn cuối cùng: Logistic Regression** (`class_weight='balanced'`).
Lý do: Recall cao, cân bằng Precision/Recall hợp lý (không suy biến), pipeline siêu nhẹ,
tốc độ suy luận nhanh nhất, phù hợp chạy trên container tài nguyên thấp.

Xem số liệu đầy đủ tại `ai-models/models/comparison.json` và `ai-models/models/metadata.json`
(được sinh tự động mỗi lần chạy `evaluate.py`).

## 5. Đóng gói Model (Model Packaging)

- File Model Pipeline: `ai-models/models/model.joblib`
- Các pipeline ứng viên: `ai-models/models/candidates/*.joblib`
- Schema cấu trúc dữ liệu: `ai-models/models/schema.json`
- Metadata mô hình: `ai-models/models/metadata.json`
- Metadata lần train: `ai-models/models/training_metadata.json`

Tái tạo toàn bộ pipeline (từ dữ liệu thô đến model đóng gói):
```bash
cd ai-models/src
python make_synthetic_dataset.py   # bỏ qua bước này nếu đã có dữ liệu Kaggle thật
python train.py                    # huấn luyện 4 model, ghi ai-models/models/comparison.json
python evaluate.py                 # chọn model tốt nhất, xuất model.joblib/schema.json/metadata.json
```

## 6. Kiến trúc hệ thống (System Architecture)

Microservices với 3 Docker container nối chung mạng `stroke-net`:

```text
┌─────────────┐   HTTP   ┌─────────────┐   HTTP   ┌─────────────┐
│ Frontend    ├─────────>│ Backend      ├─────────>│ AI Service  │
│ React/Nginx │<─────────┤ Node/Express │<─────────┤ FastAPI     │
│ Port 80     │  JSON    │ Port 8000    │  JSON    │ Port 8001   │
└─────────────┘          └──────┬───────┘          └──────┬──────┘
                                 │                         │
                                 ▼                         ▼
                        ┌─────────────────┐       ┌────────────────┐
                        │ MongoDB Atlas   │       │ model.joblib   │
                        │ (lịch sử dự đoán)│       │ (load lúc start)│
                        └─────────────────┘       └────────────────┘
```

- **Frontend:** Tạo form nhập liệu tự động từ `schema.json` (qua Backend `GET /api/schema`),
  gửi request tới BE, hiển thị kết quả xác suất và lịch sử.
- **Backend:** Validate dữ liệu đầu vào theo schema, gọi AI Service, lưu lịch sử vào MongoDB,
  log toàn bộ luồng request kèm `request_id`.
- **AI Service:** Tự động load `model.joblib` khi khởi động, cung cấp `POST /predict`, `GET /health`,
  `GET /schema` và `GET /model-info`.

## 7. Hướng dẫn chạy trên máy local (Local Setup)

**Yêu cầu:** Docker + Docker Compose, Git.

```bash
git clone https://github.com/your-username/17_12523101_10123337_DuDoanDotQuy.git
cd 17_12523101_10123337_DuDoanDotQuy
cp .env.example .env
# Mở .env, điền MONGODB_URI thật (MongoDB Atlas) trước khi chạy
docker compose up --build
```

Kiểm tra trạng thái hệ thống:
- Frontend Web App: http://localhost:3000
- Backend Health Check: http://localhost:8000/health
- AI Service Health Check: http://localhost:8001/health
- AI Service API Docs (Swagger): http://localhost:8001/docs

## 8. Hướng dẫn huấn luyện lại Model (Retraining Guide)

### Chạy bằng script trong VS Code

```bash
python ai-models/src/train.py
python ai-models/src/evaluate.py
python ai-models/src/validate_contract.py
```

### Chạy bằng notebook

Mở và chạy lần lượt `ai-models/colab/01_eda.ipynb`, `02_preprocess.ipynb`,
`03_train.ipynb`, `04_evaluate.ipynb`. Toàn bộ logic tương ứng nằm trong:
- `preprocess.py`: xử lý missing value (`bmi`), mã hóa đặc trưng, ColumnTransformer.
- `train.py`: huấn luyện 4 model + GridSearchCV.
- `evaluate.py`: so sánh metric, chọn model, đóng gói `model.joblib`/`schema.json`/`metadata.json`,
  sinh 5 hình EDA + confusion matrix vào `docs/figures/`.

Phần 03 lưu 4 pipeline ứng viên tại `ai-models/models/candidates/`, bảng so sánh tại
`ai-models/models/comparison.json` và metadata train tại `ai-models/models/training_metadata.json`.
Phần 04 chọn model cuối và cập nhật `ai-models/models/model.joblib`.

Sau khi chạy xong phần 04 trên Colab, cell export cuối tạo `stroke_models_evaluated.zip`.
Tải file này về máy, sau đó đồng bộ artifact vào repo bằng:

```bash
python ai-models/src/import_colab_artifacts.py path/to/stroke_models_evaluated.zip
```

Script sẽ cập nhật `model.joblib`, `schema.json`, `metadata.json`, `comparison.json`,
`training_metadata.json` và các pipeline trong `ai-models/models/candidates/`.

Backend cung cấp cả `GET /api/history` (endpoint chuẩn) và `GET /api/predictions` (alias tương thích).

## 9. Biến môi trường (Environment Variables)

Quản lý qua file `.env` (mẫu tại `.env.example`):

| Biến môi trường | Mặc định (Local Docker) | Giá trị khi Public / Tunnel | Ý nghĩa |
|---|---|---|---|
| `PORT_FE` | 3000 | 3000 | Cổng lắng nghe của Frontend |
| `PORT_BE` | 8000 | 8000 | Cổng lắng nghe của Backend |
| `PORT_AI` | 8001 | 8001 | Cổng lắng nghe của AI Service |
| `AI_SERVICE_URL` | http://ai-service:8001 | https://stroke-ai-service.onrender.com | Backend gọi sang AI Service |
| `API_URL` | http://localhost:8000 | https://stroke-backend.onrender.com | Frontend gọi sang Backend API |
| `MONGODB_URI` | mongodb+srv://... | mongodb+srv://... | Chuỗi kết nối CSDL MongoDB Atlas |
| `CORS_ORIGIN` | * | https://domain-frontend | Domain được phép gọi Backend |

## 10. Phương án triển khai (Deployment)

- **Frontend:** Deploy trên Vercel.
- **Backend:** Deploy Web Service dạng Docker trên Render.
- **AI Service:** Deploy Web Service dạng Docker trên Render.

Mọi kết nối sử dụng địa chỉ public thông qua cấu hình trong `.env`. Khi dùng Ngrok Tunnel:
cập nhật `.env`, ghi nhật ký vào mục 12, cập nhật link mới tại mục 11.

## 11. Demo Online (Public URLs)

> Chưa deploy trong lượt tạo repo này — điền lại khi deploy thật.

- Frontend: _(chưa deploy)_
- Backend API: _(chưa deploy)_
- AI Service OpenAPI Docs: _(chưa deploy)_

## 12. Nhật ký đổi cổng / tunnel (Port & Tunnel Change Log)

| Thời điểm (GMT+7) | Cấu phần đổi link | Địa chỉ cũ | Địa chỉ mới | Người thực hiện |
|---|---|---|---|---|
| 24/09/2026 | Khởi tạo repo (dữ liệu giả lập) | — | Local only | Claude (hỗ trợ dựng khung) |

## 13. Kết quả kiểm thử hiệu năng (Performance Test Results)

> Chưa chạy k6 load test trong lượt này (cần hệ thống đã deploy hoặc chạy Docker Compose thật).
> README gốc yêu cầu: 15 Concurrent VUs, 1 phút, chỉ tiêu p95 < 2000 ms và error rate < 1%.
> Lệnh mẫu k6 (đặt trong `app/backend/tests/load/predict.js` khi chuẩn bị chạy thật):
> ```bash
> k6 run --vus 15 --duration 1m app/backend/tests/load/predict.js
> ```

## 14. Việc còn lại trước khi nộp bài

- [ ] Thay `ai-models/data/dataset.zip` bằng dataset Kaggle thật, chạy lại `train.py` + `evaluate.py`.
- [x] Có đủ 4 notebook `.ipynb` trong `ai-models/colab/`.
- [x] Có contract check tại `ai-models/src/validate_contract.py`.
- [ ] Tạo `docs/baocao.docx` với nội dung báo cáo chính thức của nhóm.
- [ ] Tạo `docs/slide.pptx` với nội dung slide chính thức của nhóm.
- [ ] Deploy thật lên Vercel/Render, điền lại mục 11 và 12.
- [ ] Chạy k6 load test thật, điền lại mục 13.

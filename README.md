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
│   ├── baocao.doc               # tệp báo cáo hiện có; cần rà soát nội dung bản cuối
│   └── figures/                 # 8 hình phân tích dữ liệu và đánh giá model
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
| Naive Bayes | 1.00* | 0.88 | 0.06* | 0.11 | ~0.02 ms | 2.6 KB |
| SVM (RBF) | 0.78 | 0.88 | 0.18 | 0.29 | ~0.32 ms | 55 KB |
| Random Forest | 0.69 | 0.88 | 0.24 | 0.35 | ~0.08 ms | 542 KB |

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

```powershell
# Chạy tại thư mục gốc repository
Copy-Item .env.example .env
# Mở .env và điền MONGODB_URI hợp lệ để bật lưu lịch sử dự đoán.
docker compose up --build -d
docker compose ps
```

Compose khởi chạy ba dịch vụ; MongoDB không nằm trong compose nên cần MongoDB Atlas
hoặc MongoDB có thể truy cập được từ container backend. Nếu chưa cấu hình MongoDB,
API dự đoán vẫn có thể hoạt động nhưng lịch sử sẽ không được lưu.

Theo dõi log hoặc dừng các dịch vụ:
```powershell
docker compose logs -f
docker compose down
```

Kiểm tra trạng thái hệ thống:
- Frontend Web App: http://localhost:3000
- Backend Health Check: http://localhost:8000/health
- AI Service Health Check: http://localhost:8001/health
- AI Service API Docs (Swagger): http://localhost:8001/docs
- Backend API: `POST /api/predict`, `GET /api/schema`, `GET /api/model-info`,
  `GET /api/history` (alias: `GET /api/predictions`).

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
  sinh 6 hình EDA, biểu đồ so sánh model và confusion matrix vào `docs/figures/`.

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
| `MONGODB_URI` | mongodb+srv://... | mongodb+srv://... | Chuỗi kết nối CSDL MongoDB Atlas |
| `CORS_ORIGIN` | * | https://domain-frontend | Domain được phép gọi Backend |

Frontend hiện gọi API qua đường dẫn cùng host `/api`; Nginx trong Docker chuyển tiếp
request sang Backend. `API_URL` còn có trong `.env.example` để tham khảo nhưng hiện
chưa được frontend sử dụng. Khi deploy frontend tách biệt (ví dụ Vercel), cần cấu hình
proxy/rewrite tới Backend hoặc cập nhật frontend trước khi public.

## 10. Phương án triển khai (Deployment)

- **Frontend:** Có thể deploy trên Vercel sau khi cấu hình proxy/rewrite `/api` tới Backend.
- **Backend:** Deploy Web Service dạng Docker trên Render.
- **AI Service:** Deploy Web Service dạng Docker trên Render.

Khi dùng ngrok để public Frontend, Nginx chuyển tiếp `/api/` tới Backend qua mạng Docker;
không cần đổi `API_URL` trong `.env`. Ghi URL tunnel đang hoạt động tại mục 11 và lịch sử
thay đổi tại mục 12. Nếu public trực tiếp Backend hoặc AI Service, cần cấu hình tunnel riêng.

## 11. Demo Online (Public URLs)

Demo đang được public tạm thời qua ngrok tunnel tới Frontend/Nginx. Nginx chuyển tiếp
các request `/api/` tới Backend trong Docker Compose.

- Frontend: https://revolving-unthawed-arguable.ngrok-free.dev
- Backend API (schema): https://revolving-unthawed-arguable.ngrok-free.dev/api/schema
- AI Service OpenAPI Docs: chưa public riêng; chỉ truy cập local tại http://localhost:8001/docs

URL ngrok chỉ hoạt động khi Docker Compose và tiến trình `ngrok http 3000` còn chạy.
Gói ngrok miễn phí có thể hiển thị trang cảnh báo trước khi mở ứng dụng.

## 12. Nhật ký đổi cổng / tunnel (Port & Tunnel Change Log)

| Thời điểm (GMT+7) | Cấu phần đổi link | Địa chỉ cũ | Địa chỉ mới | Người thực hiện |
|---|---|---|---|---|
| 24/09/2026 | Khởi tạo repo (dữ liệu giả lập) | — | Local only | Claude (hỗ trợ dựng khung) |
| 27/09/2026 | Frontend `localhost:3000`; Backend API qua proxy `/api` | Local only | https://revolving-unthawed-arguable.ngrok-free.dev | Ngrok tunnel |

Ngrok ánh xạ `https://revolving-unthawed-arguable.ngrok-free.dev` về `http://localhost:3000`.
Backend (`8000`) và AI Service (`8001`) không được mở trực tiếp trong tunnel này. URL chỉ
còn hoạt động khi Docker Compose và tiến trình ngrok còn chạy; nếu URL thay đổi, cập nhật
mục 11 và thêm một dòng nhật ký mới thay vì ghi đè lịch sử.

## 13. Kết quả kiểm thử hiệu năng (Performance Test Results)

Kết quả chạy cục bộ ngày 27/09/2026 bằng k6 2.3.0, qua Frontend/Nginx tới
`POST /api/predict`:

| Hạng mục | Kết quả |
|---|---:|
| Virtual users | 15 |
| Thời lượng | 1 phút |
| Tổng request | 4.870 |
| Throughput | 81,0 request/giây |
| Độ trễ trung bình | 184,66 ms |
| Độ trễ p95 | 281,3 ms |
| Độ trễ tối đa | 562,94 ms |
| Request/check lỗi | 0 / 0% |
| Ngưỡng p95 < 2.000 ms | Đạt |
| Ngưỡng lỗi < 1% | Đạt |

Phép đo chạy với MongoDB Atlas chưa kết nối, nên không bao gồm độ trễ lưu lịch sử dự đoán.
Đây là kết quả trên máy local, không đại diện cho hiệu năng khi deploy public.

Kịch bản nằm tại `app/backend/tests/load/predict.js`; mặc định chạy 15 VU trong 1 phút.
Khi đã bật Docker Compose, chạy bằng Docker trên PowerShell:
```powershell
docker run --rm -v "$((Get-Location).Path.Replace('\','/'))/app/backend/tests/load:/scripts:ro" -e BASE_URL=http://host.docker.internal:3000 grafana/k6 run /scripts/predict.js
```

## 14. Việc còn lại trước khi nộp bài

- [x] Thay `ai-models/data/dataset.zip` bằng dataset Kaggle thật, chạy lại `train.py` + `evaluate.py`.
- [x] Có đủ 4 notebook `.ipynb` trong `ai-models/colab/`.
- [x] Có contract check tại `ai-models/src/validate_contract.py`.
- [x] Có tệp báo cáo `docs/baocao.doc` (rà soát nội dung và định dạng bản cuối trước khi nộp).
- [ ] Hoàn thiện slide thuyết trình và lưu tại `docs/slide.pptx`.
- [x] Public demo tạm thời qua Tunnel/Ngrok và điền mục 11, 12 (chưa phải deploy production).
- [x] Chạy k6 load test cục bộ và ghi kết quả tại mục 13.

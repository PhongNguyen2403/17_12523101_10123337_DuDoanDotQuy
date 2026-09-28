# Hệ thống Dự đoán Nguy cơ Đột quỵ (Stroke Risk Prediction)

> **Môn học:** Học máy cơ bản (221180) — Lớp 12523W.2
> **Giảng viên hướng dẫn:** Nguyễn Đức Tuấn Anh
> **Phiên bản tài liệu áp dụng:** 2.0 (23/09/2026)

## Cấu trúc Repository


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
- **Cột mục tiêu (target):** `stroke` (0: Không ghi nhận đột quỵ, 1: Có ghi nhận đột quỵ trong dữ liệu).
- **Ý nghĩa thực tế:** Cung cấp công cụ hỗ trợ cho y bác sĩ và người dùng tự kiểm tra sức khỏe.
  Do đặc thù y tế, hệ thống ưu tiên giảm thiểu tối đa Âm tính giả (False Negative - FN).

## 3. Dữ liệu (Dataset Information)

- **Nguồn dữ liệu:** Kaggle — [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (ODbL).
- **Lưu trữ repo:** `ai-models/data/dataset.zip` (kèm `DATA.md`).

> `ai-models/data/dataset.zip` hiện chứa dữ liệu thật tải từ Kaggle. Xem mô tả nguồn và
> cách script/notebook đọc dữ liệu tại `ai-models/data/DATA.md`.

Notebook đọc CSV bên trong `dataset.zip`. Script `ai-models/src/preprocess.py` ưu tiên CSV rời
nếu có, nếu không sẽ đọc từ ZIP; hãy bảo đảm hai tệp cùng nguồn dữ liệu. Hiện repo cung cấp ZIP.

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

Các model được huấn luyện trên cùng pipeline tiền xử lý, cùng Stratified Train/Test = 80/20,
GridSearchCV tối ưu theo `recall`. Vì dữ liệu mất cân bằng (~5% nhãn 1), báo cáo Recall,
Precision, F1 và ROC-AUC; không dùng riêng Accuracy để kết luận.

Các số liệu dưới đây được tạo ngày 28/09/2026 trên dữ liệu Kaggle trong
`ai-models/data/dataset.zip`, với 4.088 mẫu train và 1.022 mẫu test (stratified 80/20).

| Model | Recall (Class 1) | ROC-AUC | Precision (Class 1) | F1-Score | Suy luận/mẫu | Kích thước |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8000 | 0.8416 | 0.1342 | 0.2299 | 0.077 ms | 2.3 KB |
| Naive Bayes | 0.9800 | 0.7860 | 0.0641 | 0.1202 | 0.012 ms | 2.7 KB |
| SVM (RBF) | 0.7800 | 0.8244 | 0.1238 | 0.2137 | 0.625 ms | 59.2 KB |
| **Random Forest** | 0.7600 | 0.8254 | 0.1357 | 0.2303 | 0.092 ms | 479.0 KB |

Không model nào đạt ngưỡng Precision lớp 1 `0.15`. Vì vậy quy tắc fallback chọn F1 cao nhất,
với Recall và ROC-AUC làm tiêu chí phụ; model được đóng gói là **Random Forest**. Quy tắc này
tránh chọn Naive Bayes chỉ vì Recall cao trong khi Precision rất thấp. Random Forest có F1 cao
nhất (`0.2303`), nhưng Precision vẫn chỉ `0.1357`; đây là lựa chọn tốt nhất trong bốn model
cho lần thực nghiệm này, **không phải khuyến nghị sử dụng lâm sàng**.

Nếu có model đạt Precision tối thiểu, bộ chọn ưu tiên Recall test cao nhất rồi ROC-AUC. Nếu
không model nào đạt, bộ chọn dùng F1 test, rồi Recall và ROC-AUC. Metadata ghi lại nhánh chọn
đã dùng và trạng thái đạt ngưỡng tại `selection_rule`.

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
python make_synthetic_dataset.py   # chỉ chạy khi chủ động muốn tạo synthetic; lệnh này ghi đè dữ liệu hiện có
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

### Yêu cầu

- Windows 10/11 với Docker Desktop đã cài và đang chạy (Linux containers) hoặc Docker Engine có hỗ trợ Docker Compose.
- Git nếu cần clone repository.
- RAM trống khuyến nghị từ 4 GB để build các image.
- MongoDB Atlas hoặc MongoDB có thể truy cập từ container backend nếu muốn lưu lịch sử. MongoDB không được tạo tự động bởi Docker Compose; cấu hình này là tùy chọn.

### Bước 1: Mở đúng thư mục dự án

Mở PowerShell tại thư mục gốc repository, nơi có `docker-compose.yml`. Nếu chưa clone dự án:

```powershell
git clone <URL-repository>
cd 17_12523101_10123337_DuDoanDotQuy
```

Kiểm tra Docker và các tệp cần thiết:

```powershell
docker --version
docker compose version
Test-Path .\docker-compose.yml
Test-Path .\ai-models\models\model.joblib
Test-Path .\ai-models\models\schema.json
```

Hai lệnh `Test-Path` cuối cần trả về `True`. Docker Desktop phải đang chạy trước khi tiếp tục.

### Bước 2: Tạo cấu hình môi trường

Tạo `.env` từ file mẫu; nếu `.env` đã tồn tại, giữ nguyên để tránh ghi đè cấu hình cá nhân:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

Các cổng mặc định là Frontend `3000`, Backend `8000` và AI Service `8001`. Có thể giữ nguyên các giá trị mẫu nếu những cổng này chưa được ứng dụng khác sử dụng.

Để bật lưu lịch sử, thay `MONGODB_URI` trong `.env` bằng connection string hợp lệ của MongoDB Atlas hoặc MongoDB khác mà container backend truy cập được. Với Atlas, cần tạo database user và cho phép IP máy chạy Docker trong Network Access. Không chia sẻ hoặc commit `.env` chứa thông tin xác thực.

Nếu chưa cấu hình MongoDB, vẫn có thể chạy giao diện và dự đoán; trạng thái DB sẽ là `disconnected`, còn API lịch sử sẽ trả HTTP `503`.

### Bước 3: Build và khởi động ứng dụng

Chạy tại thư mục gốc repository:

```powershell
docker compose config --quiet
docker compose up --build -d
```

Compose build và khởi động ba dịch vụ: `ai-service`, `backend`, `frontend`. Lần build đầu cần tải các image và cài dependencies nên có thể mất vài phút. AI Service cần sẵn sàng trước khi Backend hoạt động đầy đủ.

Kiểm tra trạng thái container:

```powershell
docker compose ps
```

Đợi `ai-service`, `backend` và `frontend` ở trạng thái `Up`; health của `ai-service` cần là `healthy`. Xem log nếu container chưa sẵn sàng:

```powershell
docker compose logs --tail 100 ai-service
docker compose logs --tail 100 backend
docker compose logs --tail 100 frontend
```

### Bước 4: Kiểm tra ứng dụng

Mở giao diện tại <http://localhost:3000>. Kiểm tra các API bằng PowerShell:

```powershell
Invoke-RestMethod http://localhost:8001/health
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/api/schema
```

AI Service cần trả `status: ok` và `model_loaded: true`. Backend cần trả `status: ok`, `aiService: ok`; trường `db` là `connected` nếu đã cấu hình MongoDB, nếu không sẽ là `disconnected`. Tài liệu Swagger của AI Service ở <http://localhost:8001/docs>.

Thử gửi một yêu cầu dự đoán qua Backend:

```powershell
$body = @{
  gender = "Female"
  age = 67
  hypertension = 0
  heart_disease = 1
  ever_married = "Yes"
  work_type = "Private"
  Residence_type = "Urban"
  avg_glucose_level = 228.69
  bmi = 36.6
  smoking_status = "formerly smoked"
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/api/predict `
  -ContentType "application/json" `
  -Body $body
```

Kết quả thành công có các trường `stroke_risk_probability`, `stroke_prediction`, `risk_level` và `model_name`. Đây là đầu ra mô hình hỗ trợ tham khảo, không thay thế chẩn đoán y tế. Lịch sử nằm tại `GET http://localhost:8000/api/history` và chỉ dùng được khi MongoDB đã kết nối.

### Dừng và khởi động lại

```powershell
docker compose down
```

Lệnh trên dừng và xóa các container/network do Compose tạo, nhưng giữ nguyên image và dữ liệu MongoDB bên ngoài Compose. Khởi động lại lần sau bằng `docker compose up -d`; nếu có thay đổi mã nguồn hoặc Dockerfile, dùng `docker compose up --build -d`.

### Xử lý lỗi thường gặp

- **Cổng đã được sử dụng:** đổi `PORT_FE`, `PORT_BE` hoặc `PORT_AI` trong `.env`, rồi chạy lại `docker compose up -d`. Các URL local cần dùng cổng mới tương ứng.
- **`ai-service` không healthy hoặc model chưa load:** xem `docker compose logs ai-service`; xác nhận `ai-models/models/model.joblib`, `schema.json` và `metadata.json` tồn tại, tương thích với nhau.
- **Backend báo `aiService: unreachable`:** chờ AI Service khởi động, kiểm tra trạng thái/log, sau đó chạy `docker compose restart backend` nếu cần.
- **`db: disconnected`:** kiểm tra `MONGODB_URI`, thông tin xác thực và quyền truy cập mạng của MongoDB. Dự đoán vẫn có thể hoạt động nhưng không lưu lịch sử.
- **Cần xem log trực tiếp:** chạy `docker compose logs -f`; nhấn `Ctrl+C` chỉ thoát theo dõi log, không dừng container.

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

- [x] Thay `ai-models/data/dataset.zip` bằng dataset Kaggle thật.
- [ ] Chạy lại `train.py` + `evaluate.py` trên dữ liệu Kaggle và cập nhật model/metric; model hiện tại vẫn được metadata ghi nhận là train bằng synthetic.
- [x] Có đủ 4 notebook `.ipynb` trong `ai-models/colab/`.
- [x] Có contract check tại `ai-models/src/validate_contract.py`.
- [x] Có tệp báo cáo `docs/baocao.doc` (rà soát nội dung và định dạng bản cuối trước khi nộp).
- [x] Hoàn thiện slide thuyết trình và lưu tại `docs/slide.pptx`.
- [x] Public demo tạm thời qua Tunnel/Ngrok và điền mục 11, 12 (chưa phải deploy production).
- [x] Chạy k6 load test cục bộ và ghi kết quả tại mục 13.

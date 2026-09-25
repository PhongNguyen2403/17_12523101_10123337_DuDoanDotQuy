import React, { useEffect, useMemo, useState } from "react";
import { fetchSchema, fetchModelInfo, submitPrediction, fetchHistory } from "./api.js";

const FALLBACK_SCHEMA = {
  target: "stroke",
  features: [
    { name: "age", type: "number", label: "Tuổi", min: 0, max: 120 },
    { name: "avg_glucose_level", type: "number", label: "Mức đường huyết trung bình", min: 40, max: 300 },
    { name: "bmi", type: "number", label: "Chỉ số khối cơ thể (BMI)", min: 10, max: 100 },
    { name: "hypertension", type: "boolean", label: "Tiền sử cao huyết áp", values: [0, 1] },
    { name: "heart_disease", type: "boolean", label: "Tiền sử bệnh tim", values: [0, 1] },
    { name: "gender", type: "category", label: "Giới tính", values: ["Male", "Female", "Other"] },
    { name: "ever_married", type: "category", label: "Đã từng kết hôn", values: ["Yes", "No"] },
    { name: "work_type", type: "category", label: "Loại hình công việc", values: ["children", "Govt_job", "Never_worked", "Private", "Self-employed"] },
    { name: "Residence_type", type: "category", label: "Nơi sinh sống", values: ["Rural", "Urban"] },
    { name: "smoking_status", type: "category", label: "Tình trạng hút thuốc", values: ["formerly smoked", "never smoked", "smokes", "Unknown"] },
  ],
};

const RISK_COLOR = { "Thấp": "#2E7D6B", "Trung bình": "#C98A1E", "Cao": "#B4432E" };

function defaultValueFor(feature) {
  if (feature.type === "number") return "";
  if (feature.type === "boolean") return 0;
  return feature.values[0];
}

function FormField({ feature, value, onChange }) {
  if (feature.type === "number") {
    return (
      <label className="field">
        <span>{feature.label}</span>
        <input
          type="number"
          min={feature.min}
          max={feature.max}
          step="0.1"
          value={value}
          placeholder={feature.name === "bmi" ? "để trống nếu không rõ" : undefined}
          onChange={(e) => onChange(e.target.value === "" ? "" : Number(e.target.value))}
        />
      </label>
    );
  }
  if (feature.type === "boolean") {
    return (
      <label className="field">
        <span>{feature.label}</span>
        <select value={value} onChange={(e) => onChange(Number(e.target.value))}>
          <option value={0}>Không</option>
          <option value={1}>Có</option>
        </select>
      </label>
    );
  }
  return (
    <label className="field">
      <span>{feature.label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {feature.values.map((v) => (
          <option key={v} value={v}>{v}</option>
        ))}
      </select>
    </label>
  );
}

export default function App() {
  const [schema, setSchema] = useState(FALLBACK_SCHEMA);
  const [modelInfo, setModelInfo] = useState(null);
  const [values, setValues] = useState({});
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [tab, setTab] = useState("form");

  useEffect(() => {
    fetchSchema()
      .then((s) => setSchema(s))
      .catch(() => setSchema(FALLBACK_SCHEMA));
    fetchModelInfo().then(setModelInfo).catch(() => setModelInfo(null));
  }, []);

  useEffect(() => {
    const init = {};
    for (const f of schema.features) init[f.name] = defaultValueFor(f);
    setValues(init);
  }, [schema]);

  useEffect(() => {
    if (tab === "history") {
      fetchHistory(15).then((r) => setHistory(r.history || [])).catch(() => setHistory([]));
    }
  }, [tab]);

  const canSubmit = useMemo(() => {
    return schema.features.every((f) => {
      if (f.name === "bmi") return true;
      const v = values[f.name];
      return v !== "" && v !== undefined && v !== null;
    });
  }, [schema, values]);

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const payload = { ...values };
      if (payload.bmi === "") payload.bmi = null;
      const res = await submitPrediction(payload);
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="masthead">
        <div className="masthead-inner">
          <h1>Kiểm tra nguy cơ đột quỵ</h1>
          <p>Nhập chỉ số sinh học và thói quen sinh hoạt để nhận đánh giá xác suất nguy cơ, dựa trên model đã được chọn theo Recall và ROC-AUC.</p>
          {modelInfo?.metrics && (
            <p className="model-info">
              Model đang chạy: <strong>{modelInfo.model_name}</strong> · Recall {(modelInfo.metrics.recall_class1 * 100).toFixed(1)}% · ROC-AUC {(modelInfo.metrics.roc_auc * 100).toFixed(1)}%
            </p>
          )}
        </div>
      </header>

      <nav className="tabs">
        <button className={tab === "form" ? "active" : ""} onClick={() => setTab("form")}>Kiểm tra mới</button>
        <button className={tab === "history" ? "active" : ""} onClick={() => setTab("history")}>Lịch sử</button>
      </nav>

      {tab === "form" && (
        <main className="content">
          <form className="panel form-panel" onSubmit={handleSubmit}>
            <div className="grid">
              {schema.features.map((f) => (
                <FormField
                  key={f.name}
                  feature={f}
                  value={values[f.name] ?? defaultValueFor(f)}
                  onChange={(v) => setValues((prev) => ({ ...prev, [f.name]: v }))}
                />
              ))}
            </div>
            <button type="submit" disabled={!canSubmit || loading} className="submit-btn">
              {loading ? "Đang tính toán..." : "Đánh giá nguy cơ"}
            </button>
            {error && <p className="error-text">{error}</p>}
          </form>

          <aside className="panel result-panel">
            {!result && !loading && (
              <p className="placeholder">Kết quả đánh giá sẽ hiển thị tại đây sau khi bạn gửi thông tin.</p>
            )}
            {loading && <p className="placeholder">Đang gửi tới mô hình dự đoán...</p>}
            {result && (
              <div>
                <div className="risk-badge" style={{ borderColor: RISK_COLOR[result.risk_level] }}>
                  <span className="risk-level" style={{ color: RISK_COLOR[result.risk_level] }}>
                    Nguy cơ {result.risk_level}
                  </span>
                  <span className="risk-prob">{(result.stroke_risk_probability * 100).toFixed(1)}%</span>
                </div>
                <dl className="result-meta">
                  <div><dt>Mô hình</dt><dd>{result.model_name}</dd></div>
                  <div><dt>Thời gian suy luận</dt><dd>{result.inference_ms} ms</dd></div>
                  <div><dt>Mã yêu cầu</dt><dd className="mono">{result.request_id?.slice(0, 8)}</dd></div>
                </dl>
                <p className="disclaimer">
                  Đây là công cụ hỗ trợ tham khảo dựa trên mô hình học máy, không thay thế chẩn đoán y khoa.
                  Vui lòng tham khảo ý kiến bác sĩ nếu bạn có lo ngại về sức khỏe.
                </p>
              </div>
            )}
          </aside>
        </main>
      )}

      {tab === "history" && (
        <main className="content single">
          <div className="panel">
            {history.length === 0 && <p className="placeholder">Chưa có lịch sử dự đoán, hoặc cơ sở dữ liệu chưa kết nối.</p>}
            {history.length > 0 && (
              <table className="history-table">
                <thead>
                  <tr>
                    <th>Thời điểm</th>
                    <th>Xác suất</th>
                    <th>Mức nguy cơ</th>
                    <th>Tuổi</th>
                    <th>Model</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((h) => (
                    <tr key={h._id}>
                      <td>{new Date(h.createdAt).toLocaleString("vi-VN")}</td>
                      <td>{(h.strokeRiskProbability * 100).toFixed(1)}%</td>
                      <td style={{ color: RISK_COLOR[h.riskLevel] }}>{h.riskLevel}</td>
                      <td>{h.input?.age}</td>
                      <td>{h.modelName}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </main>
      )}

      <footer className="footer">
        Đồ án Học máy cơ bản (221180) — Nhóm 17 · Hệ thống Dự đoán Nguy cơ Đột quỵ
      </footer>
    </div>
  );
}

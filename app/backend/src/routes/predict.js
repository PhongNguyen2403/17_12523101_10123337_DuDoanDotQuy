const express = require("express");
const fetch = require("node-fetch");
const { aiServiceUrl } = require("../config");
const { getSchema, validateAgainstSchema } = require("../schemaClient");
const Prediction = require("../models/Prediction");
const { isDbConnected } = require("../db");

const router = express.Router();

router.post("/predict", async (req, res) => {
  const requestId = req.requestId;
  try {
    const schema = await getSchema();
    const errors = validateAgainstSchema(req.body, schema);
    if (errors.length > 0) {
      return res.status(400).json({ requestId, message: "Dữ liệu không hợp lệ", errors });
    }

    const aiResp = await fetch(`${aiServiceUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
      timeout: 10000,
    });

    if (!aiResp.ok) {
      const detail = await aiResp.text();
      console.error(`[${requestId}] AI Service lỗi ${aiResp.status}: ${detail}`);
      return res.status(502).json({ requestId, message: "AI Service không phản hồi hợp lệ" });
    }

    const result = await aiResp.json();

    if (isDbConnected()) {
      Prediction.create({
        requestId,
        input: req.body,
        strokeRiskProbability: result.stroke_risk_probability,
        strokePrediction: result.stroke_prediction,
        riskLevel: result.risk_level,
        modelName: result.model_name,
        inferenceMs: result.inference_ms,
      }).catch((err) => console.error(`[${requestId}] Lỗi lưu MongoDB:`, err.message));
    }

    return res.status(200).json({ requestId, ...result });
  } catch (err) {
    console.error(`[${requestId}] Lỗi /api/predict:`, err.message);
    return res.status(500).json({ requestId, message: "Lỗi hệ thống nội bộ" });
  }
});

router.get("/predictions", async (req, res) => {
  if (!isDbConnected()) {
    return res.status(503).json({ message: "Cơ sở dữ liệu chưa kết nối." });
  }
  const limit = Math.min(parseInt(req.query.limit, 10) || 20, 100);
  const history = await Prediction.find().sort({ createdAt: -1 }).limit(limit).lean();
  return res.status(200).json({ count: history.length, history });
});

router.get("/schema", async (_req, res) => {
  try {
    const schema = await getSchema();
    return res.status(200).json(schema);
  } catch (err) {
    return res.status(502).json({ message: "Không lấy được schema từ AI Service", detail: err.message });
  }
});

module.exports = router;

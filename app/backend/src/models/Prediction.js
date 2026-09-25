const mongoose = require("mongoose");

const PredictionSchema = new mongoose.Schema(
  {
    requestId: { type: String, required: true, index: true },
    input: { type: mongoose.Schema.Types.Mixed, required: true },
    strokeRiskProbability: { type: Number, required: true },
    strokePrediction: { type: Number, required: true },
    riskLevel: { type: String, required: true },
    modelName: { type: String },
    inferenceMs: { type: Number },
  },
  { timestamps: true }
);

module.exports = mongoose.model("Prediction", PredictionSchema);

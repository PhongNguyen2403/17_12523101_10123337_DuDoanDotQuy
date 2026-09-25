const express = require("express");
const cors = require("cors");
const helmet = require("helmet");
const morgan = require("morgan");
const fetch = require("node-fetch");

const { port, corsOrigin, aiServiceUrl } = require("./config");
const { connectDB, isDbConnected } = require("./db");
const requestIdMiddleware = require("./middleware/requestId");
const predictRouter = require("./routes/predict");

const app = express();

app.use(helmet());
app.use(cors({ origin: corsOrigin }));
app.use(express.json());
app.use(morgan("tiny"));
app.use(requestIdMiddleware);

app.get("/health", async (_req, res) => {
  let aiServiceStatus = "unreachable";
  try {
    const resp = await fetch(`${aiServiceUrl}/health`, { timeout: 3000 });
    aiServiceStatus = resp.ok ? "ok" : `error_${resp.status}`;
  } catch (_err) {
    aiServiceStatus = "unreachable";
  }

  res.status(200).json({
    status: "ok",
    service: "backend",
    db: isDbConnected() ? "connected" : "disconnected",
    aiService: aiServiceStatus,
  });
});

app.use("/api", predictRouter);

app.use((req, res) => {
  res.status(404).json({ requestId: req.requestId, message: "Không tìm thấy endpoint" });
});

async function start() {
  await connectDB();
  app.listen(port, () => {
    console.log(`[backend] Đang chạy tại cổng ${port}`);
  });
}

if (require.main === module) {
  start();
}

module.exports = app;

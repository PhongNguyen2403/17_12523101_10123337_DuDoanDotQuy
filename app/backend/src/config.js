require("dotenv").config();

module.exports = {
  port: process.env.PORT_BE || 8000,
  aiServiceUrl: process.env.AI_SERVICE_URL || "http://ai-service:8001",
  mongoUri: process.env.MONGODB_URI || "mongodb://localhost:27017/stroke_risk",
  corsOrigin: process.env.CORS_ORIGIN || "*",
};

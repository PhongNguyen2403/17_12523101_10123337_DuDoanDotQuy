const mongoose = require("mongoose");
const { mongoUri } = require("./config");

let isConnected = false;

async function connectDB() {
  if (isConnected) return;
  try {
    await mongoose.connect(mongoUri, { serverSelectionTimeoutMS: 8000 });
    isConnected = true;
    console.log("[db] Đã kết nối MongoDB.");
  } catch (err) {
    console.error("[db] Kết nối MongoDB thất bại:", err.message);
    // Không throw để service vẫn chạy được /health, /predict (chỉ mất tính năng lưu lịch sử)
  }
}

function isDbConnected() {
  return mongoose.connection.readyState === 1;
}

module.exports = { connectDB, isDbConnected };

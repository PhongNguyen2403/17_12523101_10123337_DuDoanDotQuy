const test = require("node:test");
const assert = require("node:assert");
const request = require("supertest");
const app = require("../src/server");

test("GET /health trả về status ok", async () => {
  const res = await request(app).get("/health");
  assert.strictEqual(res.status, 200);
  assert.strictEqual(res.body.status, "ok");
  assert.strictEqual(res.body.service, "backend");
});

test("GET /route-khong-ton-tai trả về 404", async () => {
  const res = await request(app).get("/route-khong-ton-tai");
  assert.strictEqual(res.status, 404);
});

test("POST /api/predict thiếu dữ liệu -> lỗi validate hoặc lỗi AI service (không crash)", async () => {
  const res = await request(app).post("/api/predict").send({});
  assert.ok([400, 500, 502].includes(res.status));
});

test("GET /api/history trả về lịch sử hoặc báo database chưa kết nối", async () => {
  const res = await request(app).get("/api/history");
  assert.ok([200, 503].includes(res.status));
});

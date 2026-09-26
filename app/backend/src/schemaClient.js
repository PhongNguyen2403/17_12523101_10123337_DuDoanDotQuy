const fetch = require("node-fetch");
const { aiServiceUrl } = require("./config");

let cachedSchema = null;
let lastFetchedAt = 0;
const CACHE_TTL_MS = 60_000;

async function getSchema() {
  const now = Date.now();
  if (cachedSchema && now - lastFetchedAt < CACHE_TTL_MS) {
    return cachedSchema;
  }
  const resp = await fetch(`${aiServiceUrl}/schema`, { timeout: 5000 });
  if (!resp.ok) throw new Error(`AI Service /schema trả lỗi ${resp.status}`);
  cachedSchema = await resp.json();
  lastFetchedAt = now;
  return cachedSchema;
}

function validateAgainstSchema(payload, schema) {
  const errors = [];
  const featureNames = new Set(schema.features.map((feature) => feature.name));
  for (const key of Object.keys(payload)) {
    if (!featureNames.has(key)) errors.push(`Trường không được hỗ trợ: ${key}`);
  }
  for (const feature of schema.features) {
    const value = payload[feature.name];
    if (value === undefined || value === null) {
      if (feature.name !== "bmi") {
        errors.push(`Thiếu trường bắt buộc: ${feature.name}`);
      }
      continue;
    }
    if (feature.type === "category" && !feature.values.includes(value)) {
      errors.push(`${feature.name} phải là một trong: ${feature.values.join(", ")}`);
    }
    if (feature.type === "boolean" && ![0, 1].includes(value)) {
      errors.push(`${feature.name} phải là 0 hoặc 1`);
    }
    if (feature.type === "number" && typeof value !== "number") {
      errors.push(`${feature.name} phải là số`);
    }
    if (feature.type === "number" && typeof value === "number") {
      if (feature.min !== undefined && value < feature.min) errors.push(`${feature.name} phải >= ${feature.min}`);
      if (feature.max !== undefined && value > feature.max) errors.push(`${feature.name} phải <= ${feature.max}`);
    }
  }
  return errors;
}

module.exports = { getSchema, validateAgainstSchema };

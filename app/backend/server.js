require('dotenv').config();
const express = require('express');
const cors = require('cors');
const morgan = require('morgan');
const axios = require('axios');
const { v4: uuidv4 } = require('uuid');

const app = express();
const port = process.env.PORT || 8000;
const aiServiceUrl = process.env.AI_SERVICE_URL || 'http://localhost:8001';
const allowedOrigin = process.env.CORS_ORIGIN || '*';

app.use(express.json({ limit: '1mb' }));
app.use(morgan('dev'));

if (allowedOrigin === '*') {
  app.use(cors());
} else {
  app.use(
    cors({
      origin: allowedOrigin,
      credentials: true,
    })
  );
}

app.use((req, res, next) => {
  req.requestId = req.headers['x-request-id'] || uuidv4();
  res.setHeader('X-Request-Id', req.requestId);
  next();
});

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'backend',
    request_id: req.requestId,
    ai_service_url: aiServiceUrl,
    uptime: process.uptime(),
  });
});

app.get('/api/health', (req, res) => {
  res.redirect('/health');
});

app.get('/api/schema', async (req, res) => {
  try {
    const response = await axios.get(`${aiServiceUrl}/schema`, { timeout: 10000 });
    return res.json(response.data);
  } catch (error) {
    const message = error.response?.data?.detail || error.message || 'Failed to fetch schema';
    return res.status(502).json({
      error: 'AI_SERVICE_SCHEMA_ERROR',
      detail: message,
      request_id: req.requestId,
    });
  }
});

app.get('/api/history', (req, res) => {
  res.json({
    message: 'History endpoint is available.',
    request_id: req.requestId,
    items: [],
  });
});

app.get('/api/predictions', (req, res) => {
  res.json({
    message: 'Predictions alias endpoint is available.',
    request_id: req.requestId,
    items: [],
  });
});

app.post('/api/predict', async (req, res) => {
  const payload = req.body || {};

  try {
    const response = await axios.post(`${aiServiceUrl}/predict`, payload, {
      timeout: 15000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    return res.json({
      request_id: req.requestId,
      data: response.data,
    });
  } catch (error) {
    const message = error.response?.data?.detail || error.message || 'Prediction failed';
    return res.status(502).json({
      error: 'AI_SERVICE_PREDICT_ERROR',
      detail: message,
      request_id: req.requestId,
    });
  }
});

app.use((req, res) => {
  res.status(404).json({
    error: 'NOT_FOUND',
    message: `Route ${req.originalUrl} does not exist.`,
    request_id: req.requestId,
  });
});

app.use((err, req, res, next) => {
  console.error('[UNHANDLED_ERROR]', err);
  res.status(500).json({
    error: 'INTERNAL_SERVER_ERROR',
    message: 'Unexpected backend error',
    request_id: req.requestId,
  });
});

app.listen(port, () => {
  console.log(`Backend running on http://localhost:${port}`);
  console.log(`AI service URL: ${aiServiceUrl}`);
});

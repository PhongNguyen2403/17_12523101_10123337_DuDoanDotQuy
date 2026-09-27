import http from 'k6/http';
import { check } from 'k6';
import { Rate } from 'k6/metrics';

const baseUrl = (__ENV.BASE_URL || 'http://localhost:3000').replace(/\/$/, '');
const predictionFailures = new Rate('prediction_failures');

export const options = {
  vus: 15,
  duration: '1m',
  thresholds: {
    http_req_duration: ['p(95)<2000'],
    http_req_failed: ['rate<0.01'],
    checks: ['rate>0.99'],
    prediction_failures: ['rate<0.01'],
  },
};

const payload = JSON.stringify({
  gender: 'Male',
  age: 67,
  hypertension: 0,
  heart_disease: 1,
  ever_married: 'Yes',
  work_type: 'Private',
  Residence_type: 'Urban',
  avg_glucose_level: 228.69,
  bmi: 36.6,
  smoking_status: 'formerly smoked',
});

export default function () {
  const response = http.post(`${baseUrl}/api/predict`, payload, {
    headers: { 'Content-Type': 'application/json' },
    tags: { endpoint: 'predict' },
  });

  let validPrediction = false;
  if (response.status === 200) {
    const body = response.json();
    validPrediction =
      typeof body.stroke_risk_probability === 'number' &&
      Number.isInteger(body.stroke_prediction);
  }

  const passed = check(response, {
    'prediction returns HTTP 200 with result': () => validPrediction,
  });
  predictionFailures.add(!passed);
}
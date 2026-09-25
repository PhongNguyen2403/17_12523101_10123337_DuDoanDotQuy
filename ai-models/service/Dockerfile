FROM python:3.11-slim

WORKDIR /srv

COPY ai-models/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ai-models/models ./models
COPY ai-models/service ./service

EXPOSE 8001

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

WORKDIR /srv/service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

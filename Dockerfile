FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1

WORKDIR /app

# 只装服务端依赖：向量模型与 chromadb 体积大，演示环境可挂载宿主缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY frontend/ ./frontend/
COPY data/taxonomy/ ./data/taxonomy/
COPY data/curated/ ./data/curated/

ENV PYTHONPATH=/app/src \
    APP_ENV=prod \
    LOG_JSON=true

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz')"

CMD ["uvicorn", "acm_coach.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

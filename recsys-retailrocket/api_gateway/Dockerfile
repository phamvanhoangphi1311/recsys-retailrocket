# ============================================
# SERVING IMAGE
# Chứa: FastAPI server + inference code + lightweight dependencies
# Dùng bởi: EKS, chạy 24/7 serve users
# Build: docker build -f api_gateway/Dockerfile -t recsys-serve:v1.0 .
# ============================================

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY api_gateway/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY api_gateway/ /app/api_gateway/
COPY src/id_mapper.py /app/src/id_mapper.py

WORKDIR /app

# Chạy FastAPI server trên port 8000
EXPOSE 8000
CMD ["uvicorn", "api_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
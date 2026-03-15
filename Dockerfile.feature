FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-feature.txt /app/requirements-feature.txt
RUN pip install --no-cache-dir -r /app/requirements-feature.txt

COPY src/feature_engineer/ /app/src/feature_engineer/
COPY src/data_pipeline/ /app/src/data_pipeline/

WORKDIR /app
CMD ["python", "--version"]

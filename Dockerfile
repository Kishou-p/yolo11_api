FROM python:3.12-slim

# CPU wheels by default (much smaller image). For NVIDIA GPUs build with
# --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.14.1
ARG TORCHVISION_VERSION=0.29.1

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    OPENCV_IO_MAX_IMAGE_PIXELS=40000000

# ultralytics pulls in opencv-python, which needs these libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

RUN useradd --system --uid 10001 --no-create-home app \
    && mkdir -p /app/data/models /app/data/uploads /app/data/outputs \
    && chown -R app:app /app/data

COPY app ./app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health/ready', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", \
     "--limit-concurrency", "64", \
     "--timeout-keep-alive", "5", \
     "--no-server-header"]

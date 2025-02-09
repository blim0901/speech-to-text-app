FROM --platform=$BUILDPLATFORM python:3.9-slim as builder

WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

ARG TARGETARCH

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS base-amd64
ENV TORCH_DEVICE=cuda
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y python3.9 python3.9-venv python3-pip libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

FROM python:3.9-slim AS base-arm64
ENV TORCH_DEVICE=mps
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

FROM base-${TARGETARCH}

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TRANSFORMERS_CACHE=/app/model_cache \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app
RUN python3.9 -m venv $VIRTUAL_ENV

COPY --from=builder /app/wheels /app/wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache /app/wheels/*

RUN python -c "from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor; model_name='facebook/wav2vec2-large-960h'; Wav2Vec2Processor.from_pretrained(model_name); Wav2Vec2ForCTC.from_pretrained(model_name)"

COPY asr_api.py .
RUN mkdir -p /app/model_cache && chmod 777 /app/model_cache

EXPOSE 8001

CMD ["python", "asr_api.py"]
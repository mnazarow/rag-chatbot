# RAG-приложение в контейнере (Linux-образ, запускается в Docker Desktop на Windows).
# Эмбеддинги/реранк считаются на CPU; генерация — через Ollama на хосте Windows.
FROM python:3.12-slim

# Системные зависимости парсеров и OCR:
#   ffmpeg            — аудио/видео для транскрибации
#   tesseract + rus   — OCR изображений и RAW-фото
#   libredwg-tools    — конвертация DWG в DXF (dwg2dxf)
#   antiword          — чтение старого .doc
#   p7zip-full, unar  — распаковка архивов (.7z/.rar и др.)
#   libgl1/libglib    — зависимости OpenCV/pymupdf/rawpy
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        tesseract-ocr tesseract-ocr-rus \
        libredwg-tools \
        antiword \
        p7zip-full unar \
        libgl1 libglib2.0-0 \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-сборка PyTorch (без CUDA — компактнее и работает в контейнере на Windows)
RUN pip install --no-cache-dir --upgrade pip wheel \
    && pip install --no-cache-dir torch==2.5.1 \
        --index-url https://download.pytorch.org/whl/cpu

# Python-зависимости приложения (Docker-вариант: faster-whisper вместо mlx)
COPY windows_variant/docker/requirements-docker.txt /app/requirements-docker.txt
RUN pip install --no-cache-dir -r requirements-docker.txt

# Код приложения
COPY . /app

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

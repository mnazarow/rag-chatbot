# RAG-приложение в контейнере (Ubuntu 24.04, запускается в Docker Desktop на Windows).
# Эмбеддинги/реранк считаются на CPU; генерация — через Ollama на хосте Windows.
#
# DWG-чертежи: пакета dwg2dxf нет в репозиториях Ubuntu/Debian, поэтому собираем
# libredwg из исходников отдельной стадией (multi-stage) — инструменты сборки в
# финальный образ не попадают. Сборка best-effort: при сбое образ всё равно
# собирается, просто без поддержки DWG.

# ---------- стадия 1: сборка libredwg (dwg2dxf) ----------
FROM ubuntu:24.04 AS dwgbuild
ENV DEBIAN_FRONTEND=noninteractive
ARG LIBREDWG_VERSION=0.13.3
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates pkg-config perl \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /dwgout/usr/local && \
    ( set -e; \
      curl -fsSL "https://github.com/LibreDWG/libredwg/releases/download/${LIBREDWG_VERSION}/libredwg-${LIBREDWG_VERSION}.tar.gz" -o /tmp/libredwg.tar.gz; \
      mkdir /tmp/libredwg; \
      tar -xf /tmp/libredwg.tar.gz -C /tmp/libredwg --strip-components=1; \
      cd /tmp/libredwg; \
      ./configure --disable-bindings --disable-werror --prefix=/usr/local; \
      make -j"$(nproc)"; \
      make install DESTDIR=/dwgout ) \
    || echo "libredwg build failed — DWG support skipped"

# ---------- стадия 2: приложение ----------
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

# Системные зависимости:
#   python3.12 (+venv/pip) — Ubuntu 24.04 поставляет Python 3.12
#   ffmpeg            — аудио/видео для транскрибации + кадры/фрагменты видео
#   tesseract + rus   — OCR изображений и RAW-фото
#   antiword          — чтение старого .doc
#   p7zip-full, unar  — распаковка архивов (.7z/.rar и др.)
#   libgl1/libglib    — зависимости OpenCV/pymupdf/rawpy
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
        ffmpeg \
        tesseract-ocr tesseract-ocr-rus \
        antiword \
        p7zip-full unar \
        libgl1 libglib2.0-0 \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# dwg2dxf и libredwg.so из стадии сборки (если собрались)
COPY --from=dwgbuild /dwgout/usr/local /usr/local
RUN ldconfig

# venv — чистый pip без конфликта с системным Python (PEP 668 externally-managed)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

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

# RAG-приложение в контейнере (Ubuntu 24.04, запускается в Docker Desktop на Windows).
# Эмбеддинги/реранк считаются на CPU; генерация — через Ollama на хосте Windows.
#
# DWG-чертежи: пакета dwg2dxf нет в репозиториях Ubuntu/Debian, поэтому собираем
# libredwg из исходников отдельной стадией (multi-stage) — инструменты сборки в
# финальный образ не попадают. Сборка best-effort: при сбое образ всё равно
# собирается, просто без поддержки DWG.

# ---------- стадия 1: сборка libredwg (dwg2dxf) из исходников ----------
# Собираем из git (не зависим от имени релиза). Весь вывод сборки пишем в лог,
# который копируется в финальный образ — при проблеме его видно в чеклисте.
FROM ubuntu:24.04 AS dwgbuild
ENV DEBIAN_FRONTEND=noninteractive
# python3 нужен configure для кодогенерации (даже с --disable-bindings)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates pkg-config perl python3 \
        autoconf automake libtool gperf texinfo \
    && rm -rf /var/lib/apt/lists/*
# Шаги связаны через && — любая ошибка останавливает сборку (без ложного «OK»).
# Весь вывод пишется в лог, который копируется в финальный образ для диагностики.
RUN mkdir -p /dwgout/usr/local && \
    ( git clone --depth 1 https://github.com/LibreDWG/libredwg /tmp/libredwg \
      && cd /tmp/libredwg \
      && sh autogen.sh \
      && ./configure --disable-bindings --disable-werror --prefix=/usr/local \
      && make -j"$(nproc)" \
      && make install DESTDIR=/dwgout \
      && echo "RESULT: OK — dwg2dxf собран" ) > /dwgout/libredwg-build.log 2>&1 \
    || echo "RESULT: FAILED — сборка libredwg не удалась (подробности в логе выше)" >> /dwgout/libredwg-build.log

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

# dwg2dxf и libredwg.so из стадии сборки (если собрались) + лог сборки (для диагностики)
COPY --from=dwgbuild /dwgout/usr/local /usr/local
COPY --from=dwgbuild /dwgout/libredwg-build.log /opt/libredwg-build.log
RUN ldconfig

# venv — чистый pip без конфликта с системным Python (PEP 668 externally-managed)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# PyTorch: по умолчанию CPU-сборка (компактно, работает везде). Для GPU передайте
# build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 (см.
# docker_variant/docker-compose.gpu.yml) — поставится CUDA-сборка torch.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --upgrade pip wheel \
    && pip install --no-cache-dir torch==2.5.1 \
        --index-url ${TORCH_INDEX_URL}

# Python-зависимости приложения (Docker-вариант: faster-whisper вместо mlx)
COPY windows_variant/docker/requirements-docker.txt /app/requirements-docker.txt
RUN pip install --no-cache-dir -r requirements-docker.txt

# Код приложения
COPY . /app

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

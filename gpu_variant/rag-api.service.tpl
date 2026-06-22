# systemd-юнит для API (шаблон; пути подставляет run_gpu.sh).
# Restart=always — позволяет кнопке «Перезапустить сервис» в админке работать
# (процесс завершается, systemd поднимает его заново).
[Unit]
Description=Corporate RAG API (FastAPI + vLLM)
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
User=__USER__
WorkingDirectory=__ROOT__
Environment=PYTHONUNBUFFERED=1
ExecStart=__ROOT__/.venv/bin/uvicorn app:app --host 0.0.0.0 --port __PORT__
Restart=always
RestartSec=2
# права на docker нужны для кнопок «Применить модель / реиндекс»
SupplementaryGroups=docker

[Install]
WantedBy=multi-user.target

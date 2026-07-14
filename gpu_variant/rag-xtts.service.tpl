# systemd-юнит для микросервиса XTTS (клонирование голоса) — шаблон.
# Пути/пользователь/порт подставляет скрипт установки (setup_gpu.sh/run_gpu.sh).
# Сервис живёт в ОТДЕЛЬНОМ окружении __ROOT__/.venv-xtts (coqui-tts + transformers>=4.57),
# чтобы не конфликтовать с ядром RAG (.venv, transformers==4.44.2). Приложение обращается
# к нему по HTTP через настройку XTTS_URL=http://127.0.0.1:__PORT__.
[Unit]
Description=Corporate RAG — XTTS voice-clone service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=__USER__
WorkingDirectory=__ROOT__
Environment=PYTHONUNBUFFERED=1
Environment=XTTS_HOST=127.0.0.1
Environment=XTTS_PORT=__PORT__
Environment=XTTS_USE_GPU=__GPU__
ExecStart=__ROOT__/.venv-xtts/bin/python __ROOT__/xtts_service.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target

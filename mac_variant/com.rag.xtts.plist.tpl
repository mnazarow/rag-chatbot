<?xml version="1.0" encoding="UTF-8"?>
<!--
  launchd-агент микросервиса XTTS (клонирование голоса) — шаблон; пути/порт
  подставляет setup.sh. Сервис живёт в отдельном окружении __ROOT__/.venv-xtts
  (coqui-tts), чтобы transformers>=4.57 не конфликтовал с ядром RAG (.venv,
  transformers==4.44.2). Приложение обращается по HTTP через XTTS_URL.
-->
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.rag.xtts</string>
    <key>ProgramArguments</key>
    <array>
        <string>__ROOT__/.venv-xtts/bin/python</string>
        <string>__ROOT__/xtts_service.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>__ROOT__</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>XTTS_HOST</key>
        <string>127.0.0.1</string>
        <key>XTTS_PORT</key>
        <string>__PORT__</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/rag_xtts.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/rag_xtts.err</string>
</dict>
</plist>

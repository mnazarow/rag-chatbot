<?xml version="1.0" encoding="UTF-8"?>
<!--
  launchd-агент автозапуска API (шаблон; пути подставляет deploy_mac.sh).
  KeepAlive=true перезапускает процесс при сбое и позволяет кнопке
  «Перезапустить сервис» в админке работать (процесс завершается — launchd
  поднимает заново).
-->
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.rag.api</string>
    <key>ProgramArguments</key>
    <array>
        <string>__ROOT__/.venv/bin/uvicorn</string>
        <string>app:app</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>__PORT__</string>
    </array>
    <key>WorkingDirectory</key>
    <string>__ROOT__</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/rag_api.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/rag_api.err</string>
</dict>
</plist>

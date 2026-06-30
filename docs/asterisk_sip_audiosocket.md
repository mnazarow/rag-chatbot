# Подключение голосового бота к Asterisk по SIP (через AudioSocket)

Полный пример: внешний **SIP-транк** провайдера приходит в Asterisk, Asterisk
отвечает на звонок и «бриджит» аудио в наш **AudioSocket-мост** (TCP), где бот
делает STT → ответ по базе знаний → TTS и отдаёт звук обратно.

```
                SIP/RTP                      AudioSocket (TCP, slin 8 кГц)
ТелефонPSTN ──────────► Asterisk ────────────────────────────► RAG-бот :8090
 (провайдер)   транк    (Answer + AudioSocket)                  STT→RAG→TTS
```

«Сырая» SIP-регистрация в самом приложении не используется — Asterisk выступает
медиашлюзом (надёжнее и портативнее).

---

## 0. Предпосылки

- **Asterisk 16+** (рекомендуется 18/20 LTS) с модулями `res_audiosocket.so` и
  `app_audiosocket.so`. Проверка:
  ```
  asterisk -rx "module show like audiosocket"
  ```
  Если пусто — пересоберите Asterisk с `menuselect` → *Applications* → `app_audiosocket`
  и *Resources* → `res_audiosocket`, либо возьмите сборку, где они включены.
- На сервере с ботом работают **ffmpeg**, **Whisper** (STT) и **TTS** — те же, что для
  голосовых сообщений в Телеграме.
- Сетевая доступность: Asterisk должен достучаться до `RAG_HOST:8090` (TCP).

---

## 1. Сторона бота (RAG)

В админке → группа **«Телефония (SIP/АТС)»**:

| Настройка | Значение | Примечание |
|---|---|---|
| `SIP_ENABLED` | вкл | включает мост (scope: restart) |
| `SIP_BRIDGE_HOST` | `0.0.0.0` | bind-адрес |
| `SIP_BRIDGE_PORT` | `8090` | этот порт укажем в диалплане |
| `SIP_GREETING` | приветствие | проговаривается при ответе |
| `SIP_SILENCE_MS` | `700` | пауза-тишина = конец реплики |
| `SIP_SILENCE_RMS` | `500` | порог громкости тишины |
| `SIP_MAX_UTTER_SEC` | `15` | макс. длина реплики |

После включения — кнопка **«Перезапустить мост»**.

**Docker:** опубликуйте порт моста, иначе Asterisk до него не достучится. В
`docker-compose.yml` сервиса приложения:

```yaml
    ports:
      - "8090:8090"        # AudioSocket-мост (SIP_BRIDGE_PORT)
```

---

## 2. Сторона Asterisk — `pjsip.conf`

### 2.1. Транспорт

```ini
[transport-udp]
type=transport
protocol=udp
bind=0.0.0.0:5060
; внешний адрес/сеть — если Asterisk за NAT:
;external_media_address=ВАШ_ВНЕШНИЙ_IP
;external_signaling_address=ВАШ_ВНЕШНИЙ_IP
;local_net=192.168.0.0/16
```

### 2.2. SIP-транк провайдера (регистрация логином/паролем)

Замените `sip.provider.ru`, `LOGIN`, `PASSWORD` на данные вашего SIP-оператора.

```ini
[provider-reg]
type=registration
transport=transport-udp
outbound_auth=provider-auth
server_uri=sip:sip.provider.ru
client_uri=sip:LOGIN@sip.provider.ru
retry_interval=60

[provider-auth]
type=auth
auth_type=userpass
username=LOGIN
password=PASSWORD

[provider]
type=endpoint
transport=transport-udp
context=from-trunk            ; входящие звонки попадают сюда
disallow=all
allow=alaw,ulaw              ; Asterisk перекодирует в slin для AudioSocket
outbound_auth=provider-auth
aors=provider
from_user=LOGIN
from_domain=sip.provider.ru
;direct_media=no             ; важно для медиамоста — звук идёт через Asterisk

[provider-identify]
type=identify
endpoint=provider
match=sip.provider.ru        ; распознавать входящие от провайдера

[provider]
type=aor
contact=sip:sip.provider.ru
```

### 2.3. (Опционально) софтфон для теста — внутренний номер 1001

```ini
[1001]
type=endpoint
transport=transport-udp
context=from-internal
disallow=all
allow=alaw,ulaw
auth=1001-auth
aors=1001

[1001-auth]
type=auth
auth_type=userpass
username=1001
password=СЛОЖНЫЙ_ПАРОЛЬ

[1001]
type=aor
max_contacts=1
```

Проверка регистрации транка:
```
asterisk -rx "pjsip show registrations"
asterisk -rx "pjsip show endpoints"
```

---

## 3. Диалплан — `extensions.conf`

`RAG_HOST` — адрес сервера с ботом, порт = `SIP_BRIDGE_PORT` (8090). UUID звонка
генерируем через `uuidgen` (макрос для повторного использования).

```ini
[globals]
RAG_HOST=10.0.0.5            ; ← IP/host сервера с ботом
RAG_PORT=8090

; --- общий блок: ответить и отдать звонок боту ---
[bot-answer]
exten => _X.,1,NoOp(Звонок боту RAG: ${CALLERID(num)})
 same => n,Answer()
 same => n,Set(CALLUUID=${SHELL(uuidgen | tr -d '\n')})
 same => n,AudioSocket(${CALLUUID},${RAG_HOST}:${RAG_PORT})
 same => n,Hangup()

; --- входящие с транка провайдера → сразу бот ---
[from-trunk]
exten => _X.,1,Goto(bot-answer,${EXTEN},1)
; или жёстко на конкретный DID:
;exten => 74950000000,1,Goto(bot-answer,s,1)

; --- внутренний набор: 700 = бот (для теста с софтфона 1001) ---
[from-internal]
exten => 700,1,Goto(bot-answer,${EXTEN},1)
```

Применить изменения:
```
asterisk -rx "dialplan reload"
asterisk -rx "pjsip reload"
```

> `Answer()` обязателен **до** `AudioSocket`. Если в вашей версии нет `uuidgen` в
> `SHELL`, подставьте фиксированный UUID: `Set(CALLUUID=11111111-1111-1111-1111-111111111111)`.

---

## 4. Сеть и firewall

- **TCP 8090** (или ваш `SIP_BRIDGE_PORT`) — Asterisk → RAG-бот.
- **UDP 5060** — SIP-сигнализация (Asterisk ↔ провайдер).
- **UDP RTP-диапазон** (по умолчанию `10000–20000`, см. `rtp.conf`) — голос.
- При NAT задайте `external_media_address` / `local_net` в транспорте.

---

## 5. Проверка

1. Софтфоном (1001) наберите **700** — должно произойти `Answer()`, и бот произнесёт
   приветствие.
2. В логах Asterisk при этом видно подключение AudioSocket; в админке RAG, раздел
   **«Телефония»**, растёт счётчик активных звонков.
3. Позвоните на ваш городской номер (DID) — звонок придёт через `from-trunk` к тому же боту.

Полезные команды:
```
asterisk -rvvv                          # консоль с логами
asterisk -rx "core show channels"       # активные каналы
asterisk -rx "module show like audiosocket"
```

Тонкая настройка распознавания конца реплики — параметры `SIP_SILENCE_MS` и
`SIP_SILENCE_RMS` в админке (если бот «перебивает» или долго ждёт).
```
```

"""Нативная SIP-регистрация (без AudioSocket).

Бот регистрируется как обычный SIP-аккаунт на АТС/провайдере и принимает звонки
напрямую — AudioSocket и Asterisk-диалплан не нужны. RTP-аудио (PCMU/PCMA, 8 кГц)
обрабатывается через библиотеку pyVoIP; пайплайн распознавания и ответа
(STT → RAG → TTS) — общий с модулем `sip_bridge`.

Когда использовать этот режим вместо AudioSocket:
  - в Asterisk нет модулей AudioSocket (app_audiosocket/res_audiosocket);
  - АТС/провайдер даёт обычный SIP-аккаунт (логин/пароль) — проще «зарегистрировать
    телефон», чем настраивать диалплан.

Требуется: установленный пакet `pyVoIP`, ffmpeg, рабочие Whisper (STT) и TTS,
сетевой доступ для SIP (порт 5060/UDP) и диапазона RTP-портов.
"""
from __future__ import annotations
import importlib.util
import shutil
import socket
import threading
import time

import settings

_RATE = 8000
# pyVoIP (PCMU/PCMA) работает с 8 кГц/моно/**8 бит unsigned** PCM — НЕ 16-бит!
# (см. документацию pyVoIP). Отсюда 1 байт = 1 сэмпл, 20 мс = 160 байт.
_FRAME = 160          # 20 мс при 8 кГц/8 бит unsigned/моно

_phone = None
_stop = threading.Event()
_lock = threading.Lock()
_state = {"running": False, "registered": False, "calls": 0, "active": 0, "error": None}


def _cfg(key, default=None):
    v = settings.get(key)
    return v if v not in (None, "") else default


def _guess_ip() -> str:
    """Определить локальный IP (для SDP). Пусто/ошибка → 0.0.0.0."""
    explicit = str(_cfg("SIP_LOCAL_IP", "") or "").strip()
    if explicit:
        return explicit
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((str(_cfg("SIP_SERVER", "8.8.8.8")), int(_cfg("SIP_PORT", 5060) or 5060)))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "0.0.0.0"


def _tts_u8(text: str) -> bytes:
    """Озвучить текст → PCM 8 кГц/моно/**8 бит unsigned** (формат pyVoIP для PCMU/PCMA).
    Именно unsigned 8-бит; передача 16-бит вызывает шипение/треск."""
    import os
    import subprocess
    import tempfile
    text = (text or "").strip()
    if not text:
        return b""
    ogg = tempfile.mktemp(suffix=".ogg")
    try:
        import tts
        if not tts.synthesize(text, ogg) or not os.path.exists(ogg):
            return b""
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", ogg, "-ar", str(_RATE), "-ac", "1",
             "-f", "u8", "-"], capture_output=True, timeout=120)
        return r.stdout if r.returncode == 0 else b""
    except Exception as e:
        print(f"[sip-reg] TTS→u8: {e}")
        return b""
    finally:
        try:
            os.remove(ogg)
        except Exception:
            pass


def _u8_to_s16(data: bytes) -> bytes:
    """Входящий звук pyVoIP (unsigned 8-бит) → signed 16-бит для RMS/STT (Whisper)."""
    if not data:
        return b""
    try:
        import audioop
        signed8 = audioop.bias(data, 1, -128)     # unsigned→signed 8-бит
        return audioop.lin2lin(signed8, 1, 2)     # 8-бит → 16-бит
    except Exception:
        # фолбэк без audioop: (b-128) → 16-бит
        import array
        out = array.array("h", (int((b - 128) * 256) for b in data))
        return out.tobytes()


def _play(call, pcm: bytes) -> None:
    """Проиграть PCM (8 кГц/8 бит unsigned/моно) в звонок.

    ВАЖНО: подаём небольшими кусками с точным темпом реального времени и небольшим
    опережением (LEAD). Один большой write_audio заставляет pyVoIP на каждом 20-мс
    кадре пересобирать огромный буфер (звук «замедляется» и трещит из-за underrun).
    Малые куски + постоянный небольшой запас в буфере дают ровный звук."""
    if not pcm:
        return
    try:
        from pyVoIP.VoIP import CallState
    except Exception:
        CallState = None
    CHUNK = 1600           # 200 мс при 8 кГц/8 бит (1 байт = сэмпл)
    LEAD = 0.4             # держим ~0.4 c звука в буфере pyVoIP
    start = time.time()
    written = 0.0          # сколько секунд звука уже отдано
    i = 0
    n = len(pcm)
    while i < n and not _stop.is_set():
        frame = pcm[i:i + CHUNK]
        i += CHUNK
        try:
            call.write_audio(frame)
        except Exception:
            return
        written += len(frame) / float(_RATE)
        target = start + written - LEAD      # не забегать вперёд больше, чем на LEAD
        while time.time() < target and not _stop.is_set():
            if CallState is not None:
                try:
                    if call.state != CallState.ANSWERED:
                        return
                except Exception:
                    return
            time.sleep(0.02)
    # дождаться проигрывания остатка буфера
    remain = (start + written) - time.time()
    if remain > 0:
        time.sleep(min(remain + 0.15, 10.0))


def _drain(call, seconds: float = 0.3) -> None:
    """Сбросить входящий звук (эхо собственного ответа), чтобы не принять его за реплику."""
    end = time.time() + seconds
    while time.time() < end and not _stop.is_set():
        try:
            call.read_audio(_FRAME, False)
        except Exception:
            break
        time.sleep(0.02)


def _on_call(call) -> None:
    """Колбэк pyVoIP на входящий звонок: ответить и вести диалог."""
    from sip_bridge import _stt, _answer, _rms
    try:
        from pyVoIP.VoIP import CallState, InvalidStateError
    except Exception as e:
        print(f"[sip-reg] pyVoIP недоступен в колбэке: {e}")
        return

    _state["calls"] += 1
    _state["active"] += 1
    aid = None
    try:
        import activity
        aid = activity.start("telephony", "Звонок (SIP-регистрация)", "соединение")
    except Exception:
        aid = None

    silence_ms = int(_cfg("SIP_SILENCE_MS", 700))
    silence_rms = float(_cfg("SIP_SILENCE_RMS", 500))
    max_utter = float(_cfg("SIP_MAX_UTTER_SEC", 15))
    need_silence = max(1, silence_ms // 20)

    try:
        call.answer()
        g = _cfg("SIP_GREETING",
                 "Здравствуйте! Это голосовой ассистент компании. Задайте вопрос после сигнала.")
        _play(call, _tts_u8(g))
        _drain(call, 0.2)

        buf = bytearray()          # накапливаем уже в 16-бит (для STT)
        voiced = False
        sil = 0
        started = 0.0
        while call.state == CallState.ANSWERED and not _stop.is_set():
            try:
                data = call.read_audio(_FRAME, False)   # unsigned 8-бит
            except InvalidStateError:
                break
            except Exception:
                break
            if not data:
                time.sleep(0.02)
                continue
            data = _u8_to_s16(data)                      # → 16-бит для RMS/STT
            rms = _rms(data)
            if rms >= silence_rms:
                if not voiced:
                    started = time.time()
                voiced = True
                sil = 0
                buf += data
            elif voiced:
                sil += 1
                buf += data

            too_long = voiced and (time.time() - started) > max_utter
            if voiced and (sil >= need_silence or too_long):
                pcm = bytes(buf)
                buf = bytearray()
                voiced = False
                sil = 0
                if aid is not None:
                    try:
                        import activity
                        activity.update(aid, stage="распознавание")
                    except Exception:
                        pass
                q = _stt(pcm)
                if not q:
                    continue
                if aid is not None:
                    try:
                        import activity
                        activity.update(aid, stage="ответ", detail=q[:60])
                    except Exception:
                        pass
                ans = _answer(q) or "Извините, не нашёл ответа в документах."
                _play(call, _tts_u8(ans))
                _drain(call, 0.3)
            time.sleep(0.01)
    except InvalidStateError:
        pass
    except Exception as e:
        print(f"[sip-reg] звонок: {e}")
    finally:
        _state["active"] = max(0, _state["active"] - 1)
        try:
            call.hangup()
        except Exception:
            pass
        if aid is not None:
            try:
                import activity
                activity.finish(aid, ok=True, stage="завершён")
            except Exception:
                pass


def start() -> dict:
    global _phone
    if not _cfg("SIP_REGISTER_ENABLED"):
        return {"ok": False, "msg": "SIP-регистрация выключена (SIP_REGISTER_ENABLED)"}
    if importlib.util.find_spec("pyVoIP") is None:
        _state["error"] = "не установлен пакет pyVoIP"
        return {"ok": False, "msg": _state["error"]}
    with _lock:
        if _phone is not None:
            return {"ok": True, "msg": "уже запущен"}
        server = str(_cfg("SIP_SERVER", "") or "").strip()
        if not server:
            return {"ok": False, "msg": "не задан адрес SIP-сервера (SIP_SERVER)"}
        user = str(_cfg("SIP_USERNAME", "") or "").strip()
        pwd = str(_cfg("SIP_PASSWORD", "") or "")
        port = int(_cfg("SIP_PORT", 5060) or 5060)
        myip = _guess_ip()
        sipport = int(_cfg("SIP_LOCAL_PORT", 5060) or 5060)
        rtp_lo = int(_cfg("SIP_RTP_PORT_LOW", 10000) or 10000)
        rtp_hi = int(_cfg("SIP_RTP_PORT_HIGH", 20000) or 20000)
        _stop.clear()
        # диагностика: по SIP_DEBUG включаем подробный лог pyVoIP (виден обмен REGISTER)
        if _cfg("SIP_DEBUG"):
            try:
                import logging
                lg = logging.getLogger("pyVoIP")
                lg.setLevel(logging.DEBUG)
                if not lg.handlers:
                    h = logging.StreamHandler()
                    h.setFormatter(logging.Formatter("[pyVoIP] %(message)s"))
                    lg.addHandler(h)
            except Exception:
                pass
        try:
            from pyVoIP.VoIP import VoIPPhone
            print(f"[sip-reg] REGISTER → {server}:{port} как {user} "
                  f"(myIP={myip}, локальный порт {sipport})")
            _phone = VoIPPhone(server, port, user, pwd, myIP=myip,
                               sipPort=sipport, rtpPortLow=rtp_lo, rtpPortHigh=rtp_hi,
                               callCallback=_on_call)
            _phone.start()
            _state["running"] = True
            _state["error"] = None
        except Exception as e:
            _state["error"] = str(e)[:200]
            _state["running"] = False
            _state["registered"] = False
            _phone = None
            return {"ok": False, "msg": "ошибка регистрации: " + _state["error"]}
    # ждём фактического подтверждения регистрации от pyVoIP (или ошибки), до ~6 c
    phase = "registering"
    for _ in range(30):
        phase = _phone_phase()
        if phase in ("registered", "failed", "inactive"):
            break
        time.sleep(0.2)
    _state["registered"] = (phase == "registered")
    if phase == "registered":
        _state["error"] = None
        print(f"[sip-reg] зарегистрирован как {user}@{server}:{port} (myIP={myip})")
        return {"ok": True, "msg": f"зарегистрирован как {user}@{server}:{port}"}
    _state["error"] = (f"регистрация не подтверждена (статус: {phase}). Проверьте логин/пароль, "
                       f"адрес и порт сервера, а также NAT/локальный IP для SDP.")
    print(f"[sip-reg] НЕ зарегистрирован ({phase}) как {user}@{server}:{port} (myIP={myip})")
    return {"ok": False, "msg": _state["error"]}


def _phone_phase() -> str:
    """Фактическое состояние регистрации из pyVoIP: inactive|registering|registered|
    deregistering|failed|unknown. Не полагаемся на «оптимистичный» флаг."""
    p = _phone
    if p is None:
        return "inactive"
    try:
        st = p.get_status()                      # pyVoIP.VoIP.PhoneStatus
        return getattr(st, "name", str(st)).lower()
    except Exception:
        # старые/иные версии pyVoIP без get_status — берём внутреннее поле, иначе unknown
        for attr in ("_status", "status"):
            v = getattr(p, attr, None)
            if v is not None:
                return getattr(v, "name", str(v)).lower()
        return "unknown"


def stop() -> None:
    global _phone
    _stop.set()
    with _lock:
        try:
            if _phone is not None:
                _phone.stop()
        except Exception:
            pass
        _phone = None
    _state["running"] = False
    _state["registered"] = False


def restart() -> dict:
    stop()
    time.sleep(0.5)
    return start()


def status() -> dict:
    # фактическая фаза регистрации из pyVoIP (обновляется в реальном времени)
    phase = _phone_phase()
    registered = (phase == "registered") if _phone is not None \
        else bool(_state.get("registered", False))
    return {
        "enabled": bool(_cfg("SIP_REGISTER_ENABLED")),
        "running": _state.get("running", False),
        "registered": registered,
        "phase": phase,
        "server": str(_cfg("SIP_SERVER", "") or ""),
        "username": str(_cfg("SIP_USERNAME", "") or ""),
        "port": int(_cfg("SIP_PORT", 5060) or 5060),
        "local_port": int(_cfg("SIP_LOCAL_PORT", 5060) or 5060),
        "rtp_range": [int(_cfg("SIP_RTP_PORT_LOW", 10000) or 10000),
                      int(_cfg("SIP_RTP_PORT_HIGH", 20000) or 20000)],
        "calls": _state.get("calls", 0),
        "active": _state.get("active", 0),
        "error": _state.get("error"),
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "pyvoip": importlib.util.find_spec("pyVoIP") is not None,
    }

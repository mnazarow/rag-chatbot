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
_FRAME = 320          # 20 мс при 8 кГц/16 бит/моно

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


def _play(call, pcm: bytes) -> None:
    """Проиграть PCM (8 кГц/16 бит/моно) в звонок и подождать окончания."""
    if not pcm:
        return
    try:
        call.write_audio(pcm)
    except Exception:
        return
    # дать буферу проиграться (длительность аудио + небольшой запас)
    dur = len(pcm) / float(_RATE * 2)
    end = time.time() + dur + 0.2
    while time.time() < end and not _stop.is_set():
        try:
            from pyVoIP.VoIP import CallState
            if call.state != CallState.ANSWERED:
                break
        except Exception:
            break
        time.sleep(0.02)


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
    from sip_bridge import _tts_pcm, _stt, _answer, _rms
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
        _play(call, _tts_pcm(g))
        _drain(call, 0.2)

        buf = bytearray()
        voiced = False
        sil = 0
        started = 0.0
        while call.state == CallState.ANSWERED and not _stop.is_set():
            try:
                data = call.read_audio(_FRAME, False)
            except InvalidStateError:
                break
            except Exception:
                break
            if not data:
                time.sleep(0.02)
                continue
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
                _play(call, _tts_pcm(ans))
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
        try:
            from pyVoIP.VoIP import VoIPPhone
            _phone = VoIPPhone(server, port, user, pwd, myIP=myip,
                               sipPort=sipport, rtpPortLow=rtp_lo, rtpPortHigh=rtp_hi,
                               callCallback=_on_call)
            _phone.start()
            _state["running"] = True
            _state["registered"] = True
            _state["error"] = None
        except Exception as e:
            _state["error"] = str(e)[:200]
            _state["running"] = False
            _state["registered"] = False
            _phone = None
            return {"ok": False, "msg": "ошибка регистрации: " + _state["error"]}
    print(f"[sip-reg] зарегистрирован как {user}@{server}:{port} (myIP={myip})")
    return {"ok": True, "msg": f"зарегистрирован как {user}@{server}:{port}"}


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
    return {
        "enabled": bool(_cfg("SIP_REGISTER_ENABLED")),
        "running": _state.get("running", False),
        "registered": _state.get("registered", False),
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

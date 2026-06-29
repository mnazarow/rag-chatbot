"""Голосовой мост к АТС через Asterisk AudioSocket.

Бот становится телефонным агентом: Asterisk (или АТС поверх него) по приходу звонка
бриджит аудио на этот TCP-сервис (диалплан `AudioSocket(<uuid>,host:port)`), а мы:
  входной звук (PCM 8 кГц) → детект конца реплики по тишине → STT (Whisper) →
  ответ RAG (тот же, что в чате/боте) → синтез речи (TTS) → отдаём звук обратно.

Почему AudioSocket, а не «сырой» SIP: регистрация SIP + RTP в процессе требует
нативного стека (PJSIP) и хрупка; AudioSocket — простой TCP-протокол Asterisk для
внешней обработки медиа, поэтому надёжнее и портативнее. Для АТС без AudioSocket
ставится шлюз: SIP-транк в Asterisk, дальше — AudioSocket.

Требуется: Asterisk с приложением AudioSocket, ffmpeg, рабочий Whisper (STT) и TTS.
Формат AudioSocket: кадры [тип(1)][длина(2, big-endian)][данные]; аудио — signed
linear 16-bit, 8 кГц, моно (тип 0x10); 0x01 — UUID звонка, 0x00 — отбой.
"""
from __future__ import annotations
import os
import socket
import struct
import subprocess
import tempfile
import threading
import time
import wave

import settings

try:
    import audioop  # noqa  (удалён в Python 3.13)
    _HAVE_AUDIOOP = True
except Exception:
    _HAVE_AUDIOOP = False

KIND_HANGUP = 0x00
KIND_ID = 0x01
KIND_AUDIO = 0x10

_RATE = 8000           # Гц, как у AudioSocket (slin)
_FRAME = 320           # байт = 20 мс при 8 кГц/16 бит/моно

_thread = None
_srv = None
_stop = threading.Event()
_state = {"running": False, "calls": 0, "active": 0, "error": None}


def _cfg(key, default=None):
    v = settings.get(key)
    return v if v not in (None, "") else default


def _rms(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    if _HAVE_AUDIOOP:
        try:
            return float(audioop.rms(pcm, 2))
        except Exception:
            pass
    # фолбэк без audioop
    import array
    a = array.array("h")
    a.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not a:
        return 0.0
    return (sum(x * x for x in a) / len(a)) ** 0.5


def _recvn(sock, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except Exception:
            return b""
        if not chunk:
            return b""
        buf += chunk
    return buf


def _read_msg(sock):
    """Прочитать один кадр AudioSocket → (kind, payload) или (None, None)."""
    hdr = _recvn(sock, 3)
    if len(hdr) < 3:
        return None, None
    kind = hdr[0]
    length = struct.unpack(">H", hdr[1:3])[0]
    payload = _recvn(sock, length) if length else b""
    if length and len(payload) < length:
        return None, None
    return kind, payload


def _send_audio(sock, pcm: bytes, paced: bool = True) -> None:
    """Отправить PCM (8 кГц/16 бит/моно) кадрами по 20 мс."""
    for i in range(0, len(pcm), _FRAME):
        frame = pcm[i:i + _FRAME]
        if len(frame) < _FRAME:
            frame = frame + b"\x00" * (_FRAME - len(frame))
        try:
            sock.sendall(bytes([KIND_AUDIO]) + struct.pack(">H", len(frame)) + frame)
        except Exception:
            return
        if paced and not _stop.is_set():
            time.sleep(0.02)


def _tts_pcm(text: str) -> bytes:
    """Озвучить текст и получить PCM 8 кГц/16 бит/моно (через tts + ffmpeg)."""
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
             "-f", "s16le", "-"], capture_output=True, timeout=120)
        return r.stdout if r.returncode == 0 else b""
    except Exception as e:
        print(f"[sip] TTS→PCM: {e}")
        return b""
    finally:
        try:
            os.remove(ogg)
        except Exception:
            pass


def _stt(pcm: bytes) -> str:
    """Распознать накопленный PCM (8 кГц/16 бит/моно) через Whisper."""
    if len(pcm) < _RATE:          # меньше ~0.5 с — пропускаем
        return ""
    wav = tempfile.mktemp(suffix=".wav")
    try:
        with wave.open(wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(_RATE)
            w.writeframes(pcm)
        import loaders
        return (loaders.transcribe_audio(wav) or "").strip()
    except Exception as e:
        print(f"[sip] STT: {e}")
        return ""
    finally:
        try:
            os.remove(wav)
        except Exception:
            pass


def _answer(question: str) -> str:
    try:
        import telegram_bot
        text, _src, _hits = telegram_bot._answer(question)
        return (text or "").strip()
    except Exception as e:
        print(f"[sip] ответ RAG: {e}")
        return ""


def _handle(sock) -> None:
    sock.settimeout(60)
    aid = None
    try:
        import activity
        aid = activity.start("telephony", "Звонок (SIP/АТС)", "соединение")
    except Exception:
        aid = None
    _state["calls"] += 1
    _state["active"] += 1
    silence_ms = int(_cfg("SIP_SILENCE_MS", 700))
    silence_rms = float(_cfg("SIP_SILENCE_RMS", 500))
    max_utter = float(_cfg("SIP_MAX_UTTER_SEC", 15))
    frame_ms = 20
    need_silence = max(1, silence_ms // frame_ms)

    greeted = False
    buf = bytearray()
    voiced = False
    sil = 0
    utter_started = 0.0
    try:
        while not _stop.is_set():
            kind, payload = _read_msg(sock)
            if kind is None or kind == KIND_HANGUP:
                break
            if kind == KIND_ID:
                if not greeted:
                    greeted = True
                    g = _cfg("SIP_GREETING",
                             "Здравствуйте! Это голосовой ассистент компании. Задайте вопрос после сигнала.")
                    pcm = _tts_pcm(g)
                    if pcm:
                        _send_audio(sock, pcm)
                continue
            if kind != KIND_AUDIO or not payload:
                continue
            rms = _rms(payload)
            if rms >= silence_rms:
                if not voiced:
                    utter_started = time.time()
                voiced = True
                sil = 0
                buf += payload
            elif voiced:
                sil += 1
                buf += payload

            too_long = voiced and (time.time() - utter_started) > max_utter
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
                apcm = _tts_pcm(ans)
                if apcm:
                    _send_audio(sock, apcm)
                # сбрасываем накопившееся за время ответа, чтобы не принять эхо за реплику
                sock.settimeout(0.05)
                try:
                    while True:
                        k, _p = _read_msg(sock)
                        if k is None:
                            break
                except Exception:
                    pass
                sock.settimeout(60)
    except Exception as e:
        print(f"[sip] звонок: {e}")
    finally:
        _state["active"] = max(0, _state["active"] - 1)
        try:
            sock.close()
        except Exception:
            pass
        if aid is not None:
            try:
                import activity
                activity.finish(aid, ok=True, stage="завершён")
            except Exception:
                pass


def _serve(host: str, port: int) -> None:
    global _srv
    try:
        _srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _srv.bind((host, port))
        _srv.listen(8)
        _srv.settimeout(1.0)
        _state["running"] = True
        _state["error"] = None
        print(f"[sip] AudioSocket-мост слушает {host}:{port}")
    except Exception as e:
        _state["error"] = str(e)[:200]
        _state["running"] = False
        print(f"[sip] не удалось открыть {host}:{port}: {e}")
        return
    while not _stop.is_set():
        try:
            conn, _addr = _srv.accept()
        except socket.timeout:
            continue
        except Exception:
            break
        threading.Thread(target=_handle, args=(conn,), daemon=True).start()
    try:
        _srv.close()
    except Exception:
        pass
    _state["running"] = False


def start() -> dict:
    global _thread
    if not _cfg("SIP_ENABLED"):
        return {"ok": False, "msg": "телефония выключена (SIP_ENABLED)"}
    if _thread and _thread.is_alive():
        return {"ok": True, "msg": "уже запущен"}
    host = str(_cfg("SIP_BRIDGE_HOST", "0.0.0.0"))
    port = int(_cfg("SIP_BRIDGE_PORT", 8090))
    _stop.clear()
    _thread = threading.Thread(target=_serve, args=(host, port), daemon=True)
    _thread.start()
    time.sleep(0.3)
    return {"ok": _state.get("running", False),
            "msg": f"мост на {host}:{port}" if _state.get("running")
            else ("ошибка: " + (_state.get("error") or "не запущен"))}


def stop() -> None:
    _stop.set()
    try:
        if _srv:
            _srv.close()
    except Exception:
        pass
    _state["running"] = False


def restart() -> dict:
    stop()
    time.sleep(0.5)
    return start()


def status() -> dict:
    have_ff = bool(__import__("shutil").which("ffmpeg"))
    return {"enabled": bool(_cfg("SIP_ENABLED")), "running": _state.get("running", False),
            "host": str(_cfg("SIP_BRIDGE_HOST", "0.0.0.0")),
            "port": int(_cfg("SIP_BRIDGE_PORT", 8090)),
            "calls": _state.get("calls", 0), "active": _state.get("active", 0),
            "error": _state.get("error"), "ffmpeg": have_ff,
            "audioop": _HAVE_AUDIOOP}

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
_calls_lock = threading.Lock()   # защищает счётчики calls/active (гонка потоков звонков)
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
    fd, ogg = tempfile.mkstemp(suffix=".ogg")
    os.close(fd)
    try:
        import tts
        import activity
        with activity.heavy_slot():                 # TTS — тяжёлая стадия конвейера
            ok = tts.synthesize(text, ogg)
        if not ok or not os.path.exists(ogg):
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
    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        with wave.open(wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(_RATE)
            w.writeframes(pcm)
        import loaders
        import activity
        with activity.heavy_slot():                 # STT (Whisper) — тяжёлая стадия
            return (loaders.transcribe_audio(wav) or "").strip()
    except Exception as e:
        print(f"[sip] STT: {e}")
        return ""
    finally:
        try:
            os.remove(wav)
        except Exception:
            pass


def _answer(question: str, caller: str = ""):
    """Ответ RAG + запись в журнал VoIP. Возвращает (text, rid) — rid нужен, чтобы
    потом проставить голосовую оценку/комментарий этому ответу."""
    try:
        import telegram_bot
        t0 = time.time()
        text, src, hits = telegram_bot._answer(question)
        text = (text or "").strip()
        # идентификация сотрудника по добавочному номеру (для журнала)
        ext = "".join(ch for ch in str(caller or "") if ch.isdigit())
        label = ("доб. " + ext) if ext else ""
        if ext:
            try:
                import db
                emp = db.org_find_by_ext(ext)
                if emp and emp.get("name"):
                    label = f"{emp['name']} (доб. {ext})"
            except Exception:
                pass
        rid = 0
        try:
            import db
            rid = db.log_request(question, "voip", len(hits),
                                 (hits[0].get("score", 0.0) if hits else 0.0),
                                 int((time.time() - t0) * 1000), len(text),
                                 bool(hits), src, answer=text, channel="voip",
                                 caller=ext, username=label)
        except Exception as e:
            print(f"[sip] журнал VoIP: {e}")
        return text, rid
    except Exception as e:
        print(f"[sip] ответ RAG: {e}")
        return "", 0


# слова-оценки и фраза запроса комментария для голосовой обратной связи по звонку
_FB_GOOD = ("хорошо", "отлично", "хороший", "отличный", "прекрасно", "супер",
            "класс", "замечательно", "спасибо")
_FB_BAD = ("плохо", "плоха", "плохой", "ужасно", "ужасный", "неверно",
           "неправильно", "не верно", "не правильно")
_FB_COMMENT = ("добавь коммент", "добавить коммент", "оставить коммент",
               "оставь коммент", "комментарий")


def feedback_intent(q: str) -> str:
    """Намерение реплики абонента после ответа: 'good'|'bad'|'comment'|'ask'.
    Оценкой считаем только короткие реплики (до 3 слов), чтобы не путать с вопросом."""
    ql = (q or "").lower().strip(" .!?,-")
    if any(p in ql for p in _FB_COMMENT):
        return "comment"
    words = [w.strip(".,!?") for w in ql.split()]
    if len(words) <= 3:
        if any(w in _FB_GOOD for w in words):
            return "good"
        if any(w in _FB_BAD for w in words):
            return "bad"
    return "ask"


def _handle(sock) -> None:
    sock.settimeout(60)
    aid = None
    try:
        import activity
        aid = activity.start("telephony", "Звонок (SIP/АТС)", "соединение")
    except Exception:
        aid = None
    # счётчики calls/active увеличиваются при допуске соединения в _serve (под локом)
    expected_uuid = str(_cfg("SIP_AUDIOSOCKET_UUID", "")
                        or _cfg("SIP_AUDIOSOCKET_SECRET", "") or "").strip()
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
    last_rid = 0
    await_comment = False
    try:
        while not _stop.is_set():
            kind, payload = _read_msg(sock)
            if kind is None or kind == KIND_HANGUP:
                break
            if kind == KIND_ID:
                # аутентификация: если задан ожидаемый UUID/секрет — сверяем с payload
                # кадра идентификации (AudioSocket шлёт UUID звонка). Не совпал — рвём.
                if expected_uuid:
                    try:
                        import uuid as _uuid
                        got = (str(_uuid.UUID(bytes=payload)) if len(payload) == 16
                               else payload.decode("utf-8", "ignore").strip())
                    except Exception:
                        got = payload.decode("utf-8", "ignore").strip()
                    if got.replace("-", "").lower() != expected_uuid.replace("-", "").lower():
                        print("[sip] отказ: UUID/секрет соединения не совпал")
                        break
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

                # голосовая обратная связь по предыдущему ответу
                if await_comment and last_rid:
                    try:
                        import db
                        db.set_comment(last_rid, q)
                    except Exception:
                        pass
                    await_comment = False
                    apcm = _tts_pcm("Комментарий сохранён. Спасибо.")
                    if apcm:
                        _send_audio(sock, apcm)
                    continue
                intent = feedback_intent(q) if last_rid else "ask"
                if intent in ("good", "bad"):
                    try:
                        import db
                        db.set_rating(last_rid, 1 if intent == "good" else -1)
                    except Exception:
                        pass
                    apcm = _tts_pcm("Спасибо, оценка сохранена. Скажите «добавь "
                                    "комментарий», если хотите оставить отзыв.")
                    if apcm:
                        _send_audio(sock, apcm)
                    continue
                if intent == "comment":
                    await_comment = True
                    apcm = _tts_pcm("Говорите комментарий.")
                    if apcm:
                        _send_audio(sock, apcm)
                    continue

                if aid is not None:
                    try:
                        import activity
                        activity.update(aid, stage="ответ", detail=q[:60])
                    except Exception:
                        pass
                ans, last_rid = _answer(q)
                if not ans:
                    ans = "Извините, не нашёл ответа в документах."
                if _cfg("SIP_SPEAK_ANSWER", True):
                    speak = ans
                else:
                    speak = _cfg("SIP_ACK_PHRASE", "Ваш запрос принят и записан. Спасибо.")
                apcm = _tts_pcm(speak)
                if apcm:
                    _send_audio(sock, apcm)
                # Сброс накопившегося эха ограничиваем по «стенным часам»: Asterisk шлёт
                # кадр каждые 20 мс непрерывно, поэтому без лимита цикл никогда не выйдет и
                # «съест» речь абонента. Также ловим отбой (KIND_HANGUP), чтобы не потерять его.
                drain_end = time.time() + float(_cfg("SIP_ECHO_DRAIN_SEC", 0.3) or 0.3)
                sock.settimeout(0.05)
                hung = False
                try:
                    while time.time() < drain_end and not _stop.is_set():
                        k, _p = _read_msg(sock)
                        if k is None:
                            break
                        if k == KIND_HANGUP:
                            hung = True
                            break
                except Exception:
                    pass
                sock.settimeout(60)
                if hung:
                    break
    except Exception as e:
        print(f"[sip] звонок: {e}")
    finally:
        with _calls_lock:
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
    # allowlist IP (если задан) и лимит одновременных сессий — базовая защита сервиса,
    # который принимает медиапоток без собственной аутентификации транспорта.
    allow_raw = str(_cfg("SIP_AUDIOSOCKET_ALLOW", "") or "").strip()
    allowset = {a.strip() for a in allow_raw.split(",") if a.strip()} or None
    max_sessions = int(_cfg("SIP_MAX_CONCURRENT", 8) or 8)
    while not _stop.is_set():
        try:
            conn, _addr = _srv.accept()
        except socket.timeout:
            continue
        except Exception:
            break
        peer_ip = (_addr[0] if _addr else "") or ""
        if allowset is not None and peer_ip not in allowset:
            print(f"[sip] отклонён IP {peer_ip} (нет в SIP_AUDIOSOCKET_ALLOW)")
            try:
                conn.close()
            except Exception:
                pass
            continue
        # атомарный допуск: увеличиваем active только если не превышен лимит
        with _calls_lock:
            if _state["active"] >= max_sessions:
                admit = False
            else:
                _state["active"] += 1
                _state["calls"] += 1
                admit = True
        if not admit:
            print(f"[sip] отклонён звонок: лимит одновременных сессий {max_sessions}")
            try:
                conn.close()
            except Exception:
                pass
            continue
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
    # По умолчанию слушаем ТОЛЬКО localhost (сервис без аутентификации транспорта).
    # Для доступа с Asterisk на другом хосте — задать SIP_AUDIOSOCKET_HOST/allowlist явно.
    host = str(_cfg("SIP_AUDIOSOCKET_HOST", None) or _cfg("SIP_BRIDGE_HOST", "127.0.0.1"))
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
            "host": str(_cfg("SIP_AUDIOSOCKET_HOST", None)
                        or _cfg("SIP_BRIDGE_HOST", "127.0.0.1")),
            "port": int(_cfg("SIP_BRIDGE_PORT", 8090)),
            "calls": _state.get("calls", 0), "active": _state.get("active", 0),
            "error": _state.get("error"), "ffmpeg": have_ff,
            "audioop": _HAVE_AUDIOOP}

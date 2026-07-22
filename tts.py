"""Локальный синтез речи (TTS) для голосовых ответов Телеграм-бота и VoIP.

Пробует доступные офлайн-движки в порядке предпочтения:
  1. xtts     — клонирование голоса по образцу (Coqui XTTS-v2; zero-shot, в т.ч. русский);
  2. piper    — нейросетевой TTS (нужна модель .onnx; лучшее качество, в т.ч. русский);
  3. say      — встроенный синтез macOS;
  4. espeak   — espeak-ng / espeak (Linux; «робот-голос»).
Результат конвертируется в OGG/Opus через ffmpeg — формат, который Telegram
принимает как голосовое сообщение (sendVoice). Всё локально, без облачных сервисов.

Если ни один движок (или ffmpeg) недоступен — synthesize() возвращает False, и бот
отправляет только текст.
"""
from __future__ import annotations
import os
import re
import shutil
import subprocess
import threading

import config
import settings


def _which(x):
    return shutil.which(x)


# ------------------------- Клонирование голоса (XTTS) -------------------------
# Coqui XTTS-v2: «обучение» голосового вывода на коротком образце (zero-shot).
_XTTS = None            # кэш загруженной модели (тяжёлая — грузим один раз)
_XTTS_KEY = None        # (модель, gpu) — чтобы пересоздать при смене настроек
_XTTS_LOAD_LOCK = threading.Lock()   # сериализует ленивую загрузку модели (гонка кэша)
_XTTS_SYNTH_LOCK = threading.Lock()  # tts_to_file непотокобезопасен — синтез по одному


def _xtts_importable() -> bool:
    """Установлен ли пакет Coqui TTS (import TTS)."""
    try:
        import importlib.util
        return importlib.util.find_spec("TTS") is not None
    except Exception:
        return False


def clone_sample_path() -> str:
    """Путь к текущему образцу голоса (WAV) — из настроек или дефолтный."""
    p = (settings.get("XTTS_SAMPLE") or "").strip()
    if p:
        return os.path.expanduser(p)
    # дефолт рядом с проектом
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "voice_samples", "clone.wav")


def sample_info() -> dict:
    """Сведения о текущем образце: наличие, путь, длительность (сек)."""
    p = clone_sample_path()
    if not (p and os.path.exists(p)):
        return {"exists": False, "path": p, "seconds": 0.0}
    sec = 0.0
    try:
        pr = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", p],
            capture_output=True, text=True, timeout=15)
        sec = round(float((pr.stdout or "0").strip() or 0), 1)
    except Exception:
        pass
    return {"exists": True, "path": p, "seconds": sec}


def xtts_status() -> dict:
    """Готовность клонирования голоса: сервис/пакет, образец, ffmpeg."""
    info = sample_info()
    url = (settings.get("XTTS_URL") or "").strip()
    # «установлен», если есть отдельный сервис (XTTS_URL) или пакет в этом venv
    installed = bool(url) or _xtts_importable()
    return {
        "installed": installed,
        "service_url": url,
        "sample": info,
        "ffmpeg": bool(_which("ffmpeg")),
        "ready": installed and info["exists"] and bool(_which("ffmpeg")),
        "language": (settings.get("XTTS_LANGUAGE") or "ru"),
        "gpu": bool(settings.get("XTTS_USE_GPU")),
    }


def save_voice_sample(src_path: str, dst_path: str | None = None) -> dict:
    """Нормализовать загруженный образец (любой формат) в WAV 16 кГц/моно через
    ffmpeg и сохранить как образец голоса. Возвращает {ok, path, seconds, msg}."""
    if not _which("ffmpeg"):
        return {"ok": False, "msg": "ffmpeg не найден — установите ffmpeg"}
    dst = dst_path or clone_sample_path()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    ok = _run(["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", "16000",
               "-c:a", "pcm_s16le", dst])
    if not ok or not os.path.exists(dst):
        return {"ok": False, "msg": "не удалось преобразовать образец (проверьте файл)"}
    info = sample_info()
    return {"ok": True, "path": dst, "seconds": info.get("seconds", 0.0),
            "msg": "образец сохранён"}


def _load_xtts():
    """Лениво загрузить модель XTTS (кэшируется). None, если недоступна."""
    global _XTTS, _XTTS_KEY
    model = (settings.get("XTTS_MODEL") or config.XTTS_MODEL).strip()
    use_gpu = bool(settings.get("XTTS_USE_GPU"))
    key = (model, use_gpu)
    if _XTTS is not None and _XTTS_KEY == key:
        return _XTTS
    # Лок вокруг загрузки: без него два потока могут одновременно грузить тяжёлую
    # модель (двойная память/гонка кэша). Двойная проверка внутри лока.
    with _XTTS_LOAD_LOCK:
        if _XTTS is not None and _XTTS_KEY == key:
            return _XTTS
        try:
            os.environ.setdefault("COQUI_TOS_AGREED", "1")  # без интерактивного вопроса лицензии
            from TTS.api import TTS as _CoquiTTS
            t = _CoquiTTS(model)
            try:
                t.to("cuda" if use_gpu else "cpu")
            except Exception:
                pass
            _XTTS, _XTTS_KEY = t, key
            return _XTTS
        except Exception as e:
            print(f"[tts] XTTS не загрузилась: {e}")
            return None


def _wav_to_ogg(wav: str, out_ogg: str) -> bool:
    """Конвертировать WAV → OGG/Opus (для Телеграма) и убрать временный WAV."""
    try:
        if not os.path.exists(wav):
            return False
        ok = _run(["ffmpeg", "-y", "-i", wav, "-c:a", "libopus", "-b:a", "32k", out_ogg])
        return ok and os.path.exists(out_ogg)
    finally:
        if os.path.exists(wav):
            try:
                os.remove(wav)
            except Exception:
                pass


def _synth_xtts_service(text: str, sample: str, out_ogg: str) -> bool:
    """Синтез через ОТДЕЛЬНЫЙ микросервис XTTS (свой venv, без конфликта transformers).
    XTTS_URL указывает на сервис (напр. http://127.0.0.1:8020)."""
    url = (settings.get("XTTS_URL") or "").strip()
    if not url:
        return False
    wav = out_ogg + ".wav"
    try:
        import httpx
        headers = {}
        tok = (settings.get("XTTS_TOKEN") or "").strip()
        if tok:
            headers["X-Auth-Token"] = tok
        r = httpx.post(url.rstrip("/") + "/tts", timeout=180, headers=headers, json={
            "text": text, "sample_path": sample,
            "language": (settings.get("XTTS_LANGUAGE") or "ru").strip() or "ru"})
        if r.status_code != 200:
            print(f"[tts] xtts-сервис вернул HTTP {r.status_code}: {r.text[:200]}")
            return False
        with open(wav, "wb") as f:
            f.write(r.content)
    except Exception as e:
        print(f"[tts] xtts-сервис недоступен ({url}): {e}")
        return False
    return _wav_to_ogg(wav, out_ogg)


def _synth_xtts(text: str, out_ogg: str) -> bool:
    """Синтез голосом-клоном (XTTS) → WAV → OGG/Opus.
    Приоритет — отдельный микросервис (XTTS_URL); иначе in-process (если coqui-tts стоит
    в этом же venv, что не рекомендуется из-за конфликта transformers с ядром)."""
    sample = clone_sample_path()
    if not (sample and os.path.exists(sample)):
        print("[tts] xtts: не задан образец голоса (XTTS_SAMPLE)")
        return False
    # 1) сервисный режим (рекомендуется)
    if (settings.get("XTTS_URL") or "").strip():
        return _synth_xtts_service(text, sample, out_ogg)
    # 2) in-process (fallback)
    model = _load_xtts()
    if model is None:
        return False
    wav = out_ogg + ".wav"
    try:
        # tts_to_file не потокобезопасен — синтезируем строго по одному.
        with _XTTS_SYNTH_LOCK:
            model.tts_to_file(
                text=text, speaker_wav=sample,
                language=(settings.get("XTTS_LANGUAGE") or "ru").strip() or "ru",
                file_path=wav)
    except Exception as e:
        print(f"[tts] xtts синтез не удался: {e}")
        return False
    return _wav_to_ogg(wav, out_ogg)


# Варианты голоса espeak-ng (один язык → разные тембры) — даёт «много голосов»
# на Linux-сервере без дополнительных моделей.
_ESPEAK_VARIANTS = [
    ("", "обычный"), ("+m1", "муж. 1"), ("+m2", "муж. 2"), ("+m3", "муж. 3"),
    ("+m4", "муж. 4"), ("+m5", "муж. 5"), ("+m6", "муж. 6"), ("+m7", "муж. 7"),
    ("+f1", "жен. 1"), ("+f2", "жен. 2"), ("+f3", "жен. 3"), ("+f4", "жен. 4"),
    ("+f5", "жен. 5"), ("+croak", "хриплый"), ("+whisper", "шёпот"),
]


def voices(engine: str | None = None) -> list[dict]:
    """Список доступных голосов для движка [{id, label}]. id кладётся в TTS_VOICE."""
    eng = (engine or available().get("engine") or "").strip().lower()
    out: list[dict] = []
    try:
        if eng == "say" and _which("say"):
            r = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=8)
            for line in (r.stdout or "").splitlines():
                m = re.match(r"^(.+?)\s{2,}([A-Za-z]{2}[_\-][A-Za-z]{2})", line)
                if m:
                    name = m.group(1).strip()
                    out.append({"id": name, "label": f"{name} · {m.group(2)}"})
        elif eng == "espeak":
            ex = _which("espeak-ng") or _which("espeak")
            # русский с разными тембрами — в начало
            for v, lbl in _ESPEAK_VARIANTS:
                out.append({"id": "ru" + v, "label": f"Русский · {lbl}"})
            # затем прочие языки из --voices (по одному на язык)
            seen = {"ru"}
            if ex:
                r = subprocess.run([ex, "--voices"], capture_output=True, text=True, timeout=8)
                for line in (r.stdout or "").splitlines()[1:]:
                    parts = line.split()
                    if len(parts) >= 4:
                        code = parts[1]
                        base = code.split("-")[0]
                        if base in seen:
                            continue
                        seen.add(base)
                        out.append({"id": code, "label": f"{parts[3]} ({code})"})
        elif eng == "piper":
            import glob as _glob
            dirs = []
            cur = (settings.get("TTS_VOICE") or "").strip()
            if cur:
                dirs.append(os.path.dirname(cur))
            docs = settings.get("DOCS_DIR")
            if docs:
                dirs.append(os.path.join(os.path.expanduser(docs), "piper"))
            dirs += [os.path.expanduser("~/piper"),
                     os.path.expanduser("~/.local/share/piper"),
                     "/opt/piper", "/usr/share/piper", "/models/piper"]
            seen = set()
            for d in dirs:
                if not d or not os.path.isdir(d):
                    continue
                for f in sorted(_glob.glob(os.path.join(d, "*.onnx"))):
                    if f in seen:
                        continue
                    seen.add(f)
                    out.append({"id": f, "label": os.path.basename(f)})
    except Exception as e:
        print(f"[tts] перечисление голосов: {e}")
    return out


def available() -> dict:
    """Какой TTS-движок будет использован и доступен ли ffmpeg."""
    eng = (settings.get("TTS_ENGINE") or "auto").strip().lower()
    ff = bool(_which("ffmpeg"))
    if eng == "off":
        return {"ok": False, "engine": None, "candidates": [], "ffmpeg": ff}
    cand = []
    # xtts (клонирование голоса) — доступен, если задан образец И либо настроен отдельный
    # микросервис (XTTS_URL), либо coqui-tts стоит в этом venv (in-process fallback).
    # В режиме auto используется, когда всё готово (лучшее качество + нужный голос).
    _xtts_backend = bool((settings.get("XTTS_URL") or "").strip()) or _xtts_importable()
    _xr = _xtts_backend and sample_info().get("exists")
    if eng == "xtts" or (eng == "auto" and _xr):
        cand.append("xtts")
    if eng in ("auto", "piper") and _which("piper"):
        cand.append("piper")
    if eng in ("auto", "say") and _which("say"):
        cand.append("say")
    if eng in ("auto", "espeak") and (_which("espeak-ng") or _which("espeak")):
        cand.append("espeak")
    return {"ok": bool(cand) and ff, "engine": cand[0] if cand else None,
            "candidates": cand, "ffmpeg": ff}


def _run(cmd, **kw) -> bool:
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=180, **kw)
        return p.returncode == 0
    except Exception as e:
        print(f"[tts] команда не выполнена ({cmd[0]}): {e}")
        return False


def synthesize(text: str, out_ogg: str) -> bool:
    """Озвучить text и записать в out_ogg (OGG/Opus). Возвращает True при успехе.

    Перебирает доступные движки по порядку (available()["candidates"]) до первого
    успеха — если приоритетный движок молча не справился, пробуем следующий."""
    text = (text or "").strip()
    if not text:
        return False
    text = text[:3000]  # ограничение на длину озвучки
    info = available()
    if not info["ok"]:
        return False
    cands = info.get("candidates") or ([info["engine"]] if info.get("engine") else [])
    for eng in cands:
        try:
            if _synth_one(eng, text, out_ogg):
                return True
        except Exception as e:
            print(f"[tts] движок {eng} не сработал: {e}")
    return False


def _synth_one(eng: str, text: str, out_ogg: str) -> bool:
    """Синтез конкретным движком. Возвращает True при успехе (файл out_ogg готов)."""
    if eng == "xtts":
        return _synth_xtts(text, out_ogg)
    voice = (settings.get("TTS_VOICE") or "").strip()
    tmp = None
    try:
        if eng == "piper":
            if not voice or not os.path.exists(voice):
                print("[tts] piper: не задан/не найден путь к модели .onnx (TTS_VOICE)")
                return False
            tmp = out_ogg + ".wav"
            ok = _run(["piper", "--model", voice, "--output_file", tmp],
                      input=text.encode("utf-8"))
            if not ok or not os.path.exists(tmp):
                return False
            src = tmp
        elif eng == "say":
            tmp = out_ogg + ".aiff"
            cmd = ["say", "-o", tmp]
            if voice:
                cmd += ["-v", voice]
            # '--' завершает опции: текст, начинающийся с '-', не попадёт в argv как флаг
            cmd += ["--", text]
            if not _run(cmd) or not os.path.exists(tmp):
                return False
            src = tmp
        else:  # espeak / espeak-ng
            ex = _which("espeak-ng") or _which("espeak")
            tmp = out_ogg + ".wav"
            # '--' завершает опции; текст с ведущим '-' иначе трактуется как флаг
            if not _run([ex, "-v", voice or "ru", "-w", tmp, "--", text]) \
                    or not os.path.exists(tmp):
                return False
            src = tmp
        ok = _run(["ffmpeg", "-y", "-i", src, "-c:a", "libopus", "-b:a", "32k", out_ogg])
        return ok and os.path.exists(out_ogg)
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass

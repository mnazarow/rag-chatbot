"""Локальный синтез речи (TTS) для голосовых ответов Телеграм-бота.

Пробует доступные офлайн-движки в порядке предпочтения:
  1. piper    — нейросетевой TTS (нужна модель .onnx; лучшее качество, в т.ч. русский);
  2. say      — встроенный синтез macOS;
  3. espeak   — espeak-ng / espeak (Linux; «робот-голос»).
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

import settings


def _which(x):
    return shutil.which(x)


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
    """Озвучить text и записать в out_ogg (OGG/Opus). Возвращает True при успехе."""
    text = (text or "").strip()
    if not text:
        return False
    text = text[:3000]  # ограничение на длину озвучки
    info = available()
    if not info["ok"]:
        return False
    eng = info["engine"]
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
            cmd += [text]
            if not _run(cmd) or not os.path.exists(tmp):
                return False
            src = tmp
        else:  # espeak / espeak-ng
            ex = _which("espeak-ng") or _which("espeak")
            tmp = out_ogg + ".wav"
            if not _run([ex, "-v", voice or "ru", "-w", tmp, text]) or not os.path.exists(tmp):
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

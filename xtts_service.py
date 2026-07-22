#!/usr/bin/env python3
# =============================================================================
#  Микросервис XTTS (клонирование голоса) — отдельный процесс/venv.
#
#  Зачем отдельно: coqui-tts требует transformers>=4.57, а ядро RAG
#  (FlagEmbedding/эмбеддер+реранкер) жёстко привязано к transformers==4.44.2.
#  В одном окружении они несовместимы, поэтому XTTS вынесен в собственный venv
#  (.venv-xtts) и общается с приложением по HTTP. Основное приложение обращается
#  к нему через настройку XTTS_URL (напр. http://127.0.0.1:8020).
#
#  Запуск (обычно через systemd rag-xtts / launchd, ставится скриптом установки):
#     .venv-xtts/bin/python xtts_service.py
#
#  Переменные окружения:
#     XTTS_MODEL     — модель Coqui (по умолч. multilingual xtts_v2)
#     XTTS_USE_GPU   — 1/0, синтез на CUDA (по умолч. 0)
#     XTTS_HOST      — интерфейс (по умолч. 127.0.0.1 — только локально)
#     XTTS_PORT      — порт (по умолч. 8020)
#
#  Эндпоинты:
#     GET  /health   -> {"ok": true, "loaded": bool, "model": str, "gpu": bool}
#     POST /tts      {text, sample_path, language} -> audio/wav (16-bit PCM)
# =============================================================================
import os
import tempfile
import threading

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="XTTS voice-clone service")

_MODEL = None
_MODEL_NAME = os.environ.get(
    "XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
_USE_GPU = os.environ.get("XTTS_USE_GPU", "0") == "1"
_LOAD_LOCK = threading.Lock()    # сериализует загрузку тяжёлой модели
_SYNTH_LOCK = threading.Lock()   # tts_to_file непотокобезопасен — синтез строго по одному

# Безопасность: разрешённый каталог образцов голоса, токен и лимит длины текста.
_SAMPLE_DIR = os.path.realpath(os.environ.get(
    "XTTS_SAMPLE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_samples")))
_TOKEN = (os.environ.get("XTTS_TOKEN") or "").strip()
try:
    _MAX_TEXT = int(os.environ.get("XTTS_MAX_TEXT", "3000"))
except Exception:
    _MAX_TEXT = 3000


def _sample_ok(path: str) -> bool:
    """Образец должен лежать ВНУТРИ разрешённого каталога (защита от зондирования ФС
    и подстановки произвольного WAV)."""
    try:
        rp = os.path.realpath(path)
        base = _SAMPLE_DIR + os.sep
        return (rp == _SAMPLE_DIR or rp.startswith(base)) and os.path.isfile(rp)
    except Exception:
        return False


def _load():
    """Лениво загрузить модель XTTS (тяжёлая — грузим один раз, под локом)."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _LOAD_LOCK:
        if _MODEL is not None:
            return _MODEL
        os.environ.setdefault("COQUI_TOS_AGREED", "1")  # без интерактивного вопроса лицензии
        from TTS.api import TTS as _CoquiTTS  # импорт тут — стартап без модели быстрый
        t = _CoquiTTS(_MODEL_NAME)
        try:
            t.to("cuda" if _USE_GPU else "cpu")
        except Exception:
            pass
        _MODEL = t
        return _MODEL


class TtsReq(BaseModel):
    text: str
    sample_path: str          # путь к образцу голоса (WAV) — общая ФС с приложением
    language: str = "ru"


@app.get("/health")
def health():
    return {"ok": True, "loaded": _MODEL is not None,
            "model": _MODEL_NAME, "gpu": _USE_GPU}


@app.post("/tts")
def tts(r: TtsReq, x_auth_token: str = Header(default="")):
    if _TOKEN and (x_auth_token or "").strip() != _TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")
    text = (r.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")
    if len(text) > _MAX_TEXT:
        text = text[:_MAX_TEXT]
    if not (r.sample_path and _sample_ok(r.sample_path)):
        raise HTTPException(status_code=400, detail="sample_path not allowed or missing")
    fd, out = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        try:
            # синтез строго по одному (tts_to_file непотокобезопасен)
            with _SYNTH_LOCK:
                _load().tts_to_file(
                    text=text, speaker_wav=os.path.realpath(r.sample_path),
                    language=(r.language or "ru").strip() or "ru", file_path=out)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"synthesis failed: {e}")
        if not os.path.exists(out):
            raise HTTPException(status_code=500, detail="no output produced")
        with open(out, "rb") as f:
            data = f.read()
        return Response(content=data, media_type="audio/wav")
    finally:
        try:
            os.remove(out)
        except Exception:
            pass


if __name__ == "__main__":
    host = os.environ.get("XTTS_HOST", "127.0.0.1")
    port = int(os.environ.get("XTTS_PORT", "8020"))
    uvicorn.run(app, host=host, port=port)

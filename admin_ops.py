"""Операции администратора, вызываемые из веб-панели:

  - status()    : доступность Qdrant и LLM, число чанков, статус индексации
  - reindex()   : запуск переиндексации (фоновый процесс ingest.py)
  - apply_llm() : перезапуск контейнера vLLM с текущей моделью (GPU-вариант)
  - restart()   : перезапуск самого сервиса (через systemd Restart=always)

Все вызовы защищены токеном на уровне app.py. Лёгкие зависимости (httpx,
subprocess) — без импорта тяжёлых ML-библиотек.
"""
from __future__ import annotations
import hashlib
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import importlib.util as _iu
import json as _json
import shutil

import httpx

import settings
import db
import backup
import fsutil

ROOT = Path(__file__).resolve().parent

_job = {"running": False, "started": None, "finished": None, "ok": None, "log": "", "summary": "", "logfile": ""}
_ft_job = {"running": False, "started": None, "finished": None, "ok": None, "log": "", "summary": "", "logfile": ""}
_graph_job = {"running": False, "started": None, "finished": None, "ok": None, "log": "", "summary": "", "logfile": ""}
_pull_job = {"running": False, "started": None, "finished": None, "ok": None,
             "log": "", "model": "", "status": "", "percent": 0,
             "completed": 0, "total": 0, "speed": 0}
_dep_job = {"running": False, "started": None, "finished": None, "ok": None,
            "log": "", "label": "", "logfile": ""}
_web_job = {"running": False, "started": None, "finished": None, "ok": None,
            "log": "", "summary": "", "logfile": ""}
_test_job = {"running": False, "started": None, "finished": None, "ok": None,
             "log": "", "logfile": "", "results": []}
_bench_job = {"running": False, "started": None, "finished": None, "ok": None,
              "log": "", "logfile": "", "results": []}
_backup_job = {"running": False, "started": None, "finished": None, "ok": None,
               "log": "", "label": "", "result": {}}
_restore_job = {"running": False, "started": None, "finished": None, "ok": None,
                "log": "", "result": {}}
_check_job = {"running": False, "started": None, "finished": None, "ok": None,
              "log": "", "logfile": "", "results": {}}

_AV_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".mp4", ".mov", ".mkv", ".webm"}
_UNSUPPORTED_FIX = {
    ".doc": "Сконвертируйте в .docx (Word → «Сохранить как») или установите LibreOffice.",
    ".rtf": "Сконвертируйте в .docx или .txt.",
    ".odt": "Сконвертируйте в .docx.",
    ".pages": "Apple Pages не читается — экспортируйте в PDF/DOCX.",
    ".numbers": "Apple Numbers не читается — экспортируйте в XLSX/CSV.",
    ".key": "Apple Keynote не читается — экспортируйте в PDF/PPTX.",
    ".xlsb": "Бинарный Excel — сохраните как .xlsx или установите pyxlsb.",
    ".epub": "Сконвертируйте в PDF/TXT.",
    ".fb2": "Сконвертируйте в TXT/PDF.",
    ".djvu": "Сконвертируйте в PDF.",
}
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp", ".jfif"}
# OCR-форматы (картинки и RAW-фото): индексируются, но при проверке каталога
# их не парсим поштучно — OCR слишком долгий для тысяч файлов
_OCR_EXTS = _IMG_EXTS | {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf",
                         ".rw2", ".orf", ".sr2"}
# Архивы: индексируются (распаковкой), но при проверке не распаковываем — долго
_ARCHIVE_EXTS = {".zip", ".rar", ".7z", ".tar", ".gz", ".tgz", ".bz2"}
# Спец-инструменты извлечения текста (CAD, 3D-обмен, старый .doc, письма, архивы)
_CAD_EXTS = {".dxf", ".dwg", ".stp", ".step", ".igs", ".iges"}
_TOOL_EXTS = _CAD_EXTS | _ARCHIVE_EXTS | {".doc", ".msg"}


def _file_method(ext: str) -> str:
    """Как из файла извлекается текст: транскрибация / OCR / спец-инструмент / прямой."""
    ext = (ext or "").lower()
    if not ext.startswith("."):
        ext = "." + ext
    if ext in _AV_EXTS:
        return "transcribed"   # аудио/видео → Whisper
    if ext in _OCR_EXTS:
        return "ocr"           # изображения/RAW-фото → OCR
    if ext in _TOOL_EXTS:
        return "tool"          # DWG/STEP/IGES/.doc/.msg/архивы → спец-парсеры
    return "text"              # PDF/DOCX/XLSX/… → прямое извлечение текста


def _fix_for(ext: str, context: str) -> str:
    ext = (ext or "").lower()
    c = (context or "").lower()
    if "xlrd" in c:
        return "Установите xlrd: pip install \"xlrd>=2.0.1\", затем переиндексируйте."
    if "password" in c or "encrypt" in c or "decrypt" in c:
        return "Снимите пароль/шифрование с файла и сохраните заново."
    if context == "unsupported":
        if ext in _IMG_EXTS:
            return "Изображение без текста не индексируется; для сканов примените OCR и сохраните как PDF/текст."
        return _UNSUPPORTED_FIX.get(ext, f"Формат {ext or 'без расширения'} не поддерживается — сконвертируйте в PDF/DOCX/XLSX/TXT.")
    if context == "empty":
        return "Не удалось извлечь текст (возможно скан/картинки внутри) — примените OCR или проверьте содержимое."
    return "См. текст ошибки; при необходимости сконвертируйте файл в поддерживаемый формат."


def check_data_dir() -> dict:
    """Проверить весь каталог документов: проблемные файлы и неподдерживаемые типы."""
    if _check_job["running"]:
        return {"ok": False, "msg": "проверка уже идёт"}
    docs = Path(settings.get("DOCS_DIR")).expanduser()
    if not docs.exists():
        return {"ok": False, "msg": f"папка не найдена: {docs}"}
    logfile = "/tmp/rag_check.log"

    def run():
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial
        import loaders
        _check_job.update(running=True, started=time.time(), finished=None, ok=None,
                          log="", logfile=logfile, results={})
        counts = {"total": 0, "ok": 0, "empty": 0, "unsupported": 0,
                  "failed": 0, "media": 0, "timeout": 0}
        problems, unsupported = [], {}
        to_parse = []  # (rel, abspath, ext) — файлы, которые надо разобрать

        def _snapshot():
            unsup = [{"ext": (k.lstrip(".") or "(без расширения)"), "count": v,
                      "fix": _fix_for(k, "unsupported")}
                     for k, v in sorted(unsupported.items(), key=lambda x: -x[1])]
            _check_job["results"] = {"counts": dict(counts), "problems": problems[:300],
                                     "problems_total": len(problems), "unsupported": unsup}

        with open(logfile, "w", buffering=1, errors="ignore") as fp:
            fp.write(f"=== Проверка каталога: {docs} ===\n")

            # недоступные папки (Errno 5 на сетевой/битой шаре и т.п.) не срывают
            # проверку — пропускаются с пометкой в проблемах
            def _werr(e):
                wp = getattr(e, "filename", "") or str(e)
                fp.write(f"  ! недоступный путь пропущен: {wp} ({e})\n")
                problems.append({"path": wp, "ext": "",
                                 "issue": f"папка недоступна (ввод/вывод): {e}",
                                 "fix": "проверьте носитель/сеть/права доступа к папке"})

            # 1) быстрый обход и классификация (без парсинга), устойчивый к ошибкам I/O
            for p in fsutil.walk_files(docs, onerror=_werr):
                counts["total"] += 1
                rel = str(p.relative_to(docs))
                ext = p.suffix.lower()
                try:
                    sz = p.stat().st_size
                except Exception:
                    sz = 0
                if sz == 0:
                    counts["empty"] += 1
                    problems.append({"path": rel, "ext": ext.lstrip("."),
                                     "issue": "пустой файл (0 байт)", "fix": "удалите или замените файл"})
                elif ext not in _SUPPORTED:
                    counts["unsupported"] += 1
                    unsupported[ext] = unsupported.get(ext, 0) + 1
                elif ext in _AV_EXTS or ext in _OCR_EXTS or ext in _ARCHIVE_EXTS:
                    counts["media"] += 1  # медиа/OCR/архивы не парсим при проверке
                else:
                    to_parse.append((rel, str(p), ext))
            _snapshot()  # обход завершён — показываем первичную статистику

            # обработка одного результата разбора
            def _apply(rel, ext, status, issue):
                if status == "ok":
                    counts["ok"] += 1
                elif status == "timeout":
                    counts["timeout"] += 1
                    problems.append({"path": rel, "ext": ext.lstrip("."),
                                     "issue": issue or "таймаут",
                                     "fix": "большой/сложный файл — увеличьте FILE_PARSE_TIMEOUT или исключите его"})
                    fp.write(f"  ⏱ {rel}: {issue}\n")
                else:
                    counts["failed"] += 1
                    problems.append({"path": rel, "ext": ext.lstrip("."),
                                     "issue": issue or "текст не извлечён",
                                     "fix": _fix_for(ext, issue or "empty")})
                    if issue and issue != "текст не извлечён":
                        fp.write(f"  ! {rel}: {issue}\n")

            # 2) ПАРАЛЛЕЛЬНЫЙ разбор файлов в пуле процессов (узкое место — парсинг)
            timeout = int(settings.get("FILE_PARSE_TIMEOUT") or 0)
            workers = max(2, min(8, (os.cpu_count() or 4)))
            fp.write(f"Найдено {counts['total']} файлов; на разбор {len(to_parse)} "
                     f"в {workers} процессах (таймаут на файл: {timeout or '—'} c)\n")
            fp.flush()
            done = 0
            fn = partial(loaders.probe_file, timeout=timeout)
            paths = [pp for (_r, pp, _e) in to_parse]
            try:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    for (rel, _pp, ext), (status, issue) in zip(
                            to_parse, ex.map(fn, paths, chunksize=8)):
                        _apply(rel, ext, status, issue)
                        done += 1
                        if done % 100 == 0:
                            fp.write(f"  …разобрано {done}/{len(to_parse)}\n")
                            fp.flush()
                            _snapshot()
            except Exception as e:
                # фолбэк: если пул процессов недоступен — дораскатываем последовательно
                fp.write(f"  ~ параллельный режим недоступен ({e}); продолжаю последовательно\n")
                for (rel, pp, ext) in to_parse[done:]:
                    status, issue = loaders.probe_file(pp, timeout)
                    _apply(rel, ext, status, issue)
                    done += 1

            unsup = [{"ext": (k.lstrip(".") or "(без расширения)"), "count": v,
                      "fix": _fix_for(k, "unsupported")}
                     for k, v in sorted(unsupported.items(), key=lambda x: -x[1])]
            fp.write(f"\nИтог: всего {counts['total']}, ок {counts['ok']}, медиа {counts['media']}, "
                     f"пустых {counts['empty']}, неподдерж. {counts['unsupported']}, "
                     f"таймаут {counts['timeout']}, ошибок {counts['failed']}\n")
        _check_job["results"] = {"counts": counts, "problems": problems[:300],
                                 "problems_total": len(problems), "unsupported": unsup}
        _check_job["log"] = _tail(logfile)
        _check_job["ok"] = True
        _check_job["running"] = False
        _check_job["finished"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True, "msg": "проверка каталога запущена"}


def _tail(path: str, n: int = 6000) -> str:
    try:
        with open(path, "r", errors="ignore") as f:
            return f.read()[-n:]
    except Exception:
        return ""


def _bg(job: dict, label: str, cmds: list, logfile: str, timeout: int = 24 * 3600) -> dict:
    """Запустить команды последовательно в фоне, вывод — в logfile (живой лог).
    cmds: список команд (каждая — list аргументов)."""
    if job["running"]:
        return {"ok": False, "msg": f"{label}: задача уже идёт"}

    def run():
        job.update(running=True, started=time.time(), finished=None, ok=None,
                   log="", logfile=logfile)
        if "summary" in job:
            job["summary"] = ""
        if "label" in job:
            job["label"] = label
        ok = True
        try:
            with open(logfile, "w", buffering=1, errors="ignore") as fp:
                for cmd in cmds:
                    fp.write("$ " + " ".join(str(c) for c in cmd) + "\n")
                    fp.flush()
                    rc = subprocess.Popen(cmd, cwd=ROOT, stdout=fp,
                                          stderr=subprocess.STDOUT).wait(timeout=timeout)
                    if rc != 0:
                        ok = False
                        break
            job["log"] = _tail(logfile)
            if "summary" in job:
                job["summary"] = _extract_summary(job["log"])
            job["ok"] = ok
        except Exception as e:
            job["ok"] = False
            job["log"] = (_tail(logfile) + "\n" + str(e)).strip()
        job["running"] = False
        job["finished"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True, "msg": f"{label}: запущено"}

# зависимости LightRAG (как в lightrag_variant/requirements-lightrag.txt)
_LIGHTRAG_DEPS = ["lightrag-hku==1.3.0", "nano-vectordb==0.0.4.3",
                  "tiktoken==0.8.0", "networkx==3.4.2"]

# поддерживаемые типы — для подсказки «сколько документов в папке»
_SUPPORTED = {".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".xlsm", ".xls", ".csv",
              ".txt", ".md", ".html", ".htm", ".mhtml", ".mht",
              ".xml", ".json", ".url", ".msg", ".svg",
              ".dxf", ".dwg", ".stp", ".step", ".igs", ".iges",
              ".zip", ".rar", ".7z", ".tar", ".gz", ".tgz", ".bz2",
              ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".jfif",
              ".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".rw2", ".orf", ".sr2",
              ".mp3", ".wav", ".m4a", ".aac", ".mp4", ".mov", ".mkv", ".webm"}


def status() -> dict:
    out: dict = {}
    # Qdrant + число чанков через REST
    try:
        coll = settings.get("QDRANT_COLLECTION")
        r = httpx.get(f"{settings.get('QDRANT_URL')}/collections/{coll}", timeout=3)
        out["qdrant"] = r.status_code == 200
        out["chunks"] = (r.json().get("result", {}) or {}).get("points_count", 0) \
            if r.status_code == 200 else 0
    except Exception:
        out["qdrant"] = False
        out["chunks"] = 0
    # LLM
    try:
        if settings.get("LLM_BACKEND") == "openai":
            r = httpx.get(
                f"{settings.get('LLM_BASE_URL')}/models",
                headers={"Authorization": f"Bearer {settings.get('LLM_API_KEY')}"},
                timeout=3,
            )
        else:
            r = httpx.get(f"{settings.get('OLLAMA_URL')}/api/tags", timeout=3)
        out["llm"] = r.status_code == 200
    except Exception:
        out["llm"] = False
    out["backend"] = settings.get("LLM_BACKEND")

    def _jobview(jb):
        d = dict(jb)
        # живой лог: пока задача идёт, читаем хвост её logfile
        if d.get("running") and d.get("logfile"):
            d["log"] = _tail(d["logfile"])
        return d

    out["index_job"] = _jobview(_job)
    out["finetune_job"] = _jobview(_ft_job)
    out["adapter_ready"] = (ROOT / "finetune" / "adapter").exists()
    out["use_finetuned"] = bool(settings.get("USE_FINETUNED"))
    out["graph_job"] = _jobview(_graph_job)
    out["graph_ready"] = (ROOT / "graph_storage").exists()
    out["engine"] = settings.get("ENGINE")
    out["pull_job"] = dict(_pull_job)
    out["dep_job"] = _jobview(_dep_job)
    out["web_job"] = _jobview(_web_job)
    out["test_job"] = _jobview(_test_job)
    out["bench_job"] = _jobview(_bench_job)
    out["check_job"] = _jobview(_check_job)
    out["backup_job"] = dict(_backup_job)
    out["restore_job"] = dict(_restore_job)
    return out


# ============================ бенчмарк компонентов ============================
def _bench_embed():
    from retriever import _embedder
    texts = [f"строка для эмбеддинга номер {i} с небольшим текстом" for i in range(16)]
    t = time.time()
    _embedder().encode(texts, normalize_embeddings=True)
    dt = (time.time() - t) * 1000
    return "Эмбеддер (bge-m3)", dt, f"16 текстов · {dt/16:.1f} мс/текст · {16/(dt/1000):.0f} текст/с"


def _bench_rerank():
    from retriever import _reranker
    pairs = [["тестовый вопрос", f"кандидатный документ номер {i}"] for i in range(16)]
    t = time.time()
    _reranker().compute_score(pairs, normalize=True)
    dt = (time.time() - t) * 1000
    return "Реранкер (bge-reranker)", dt, f"16 пар · {dt/16:.1f} мс/пара"


def _bench_search():
    from retriever import search
    N = 5
    t = time.time()
    for _ in range(N):
        search("тестовый вопрос для замера поиска")
    dt = (time.time() - t) * 1000
    return "Поиск + реранк (search)", dt / N, f"среднее по {N} запросам"


def _bench_qdrant():
    coll = settings.get("QDRANT_COLLECTION")
    t = time.time()
    httpx.get(f"{settings.get('QDRANT_URL')}/collections/{coll}", timeout=10)
    return "Qdrant (REST)", (time.time() - t) * 1000, "запрос метаданных коллекции"


def _bench_llm():
    b, m = settings.get("LLM_BACKEND"), settings.get("LLM_MODEL")
    if b == "openai":
        t = time.time()
        r = httpx.post(f"{settings.get('LLM_BASE_URL')}/chat/completions",
                       headers={"Authorization": f"Bearer {settings.get('LLM_API_KEY')}"},
                       json={"model": m, "messages": [{"role": "user", "content": "Напиши короткий абзац о тестировании."}],
                             "max_tokens": 64, "temperature": 0}, timeout=120)
        dt = time.time() - t
        ct = (r.json().get("usage", {}) or {}).get("completion_tokens", 0)
        return "LLM генерация", dt * 1000, f"{m}: {ct} токенов · {ct/dt if dt else 0:.1f} ток/с"
    r = httpx.post(f"{settings.get('OLLAMA_URL')}/api/generate",
                   json={"model": m, "prompt": "Напиши короткий абзац о тестировании.",
                         "stream": False, "options": {"num_predict": 64}}, timeout=180)
    j = r.json()
    ec = j.get("eval_count", 0)
    ed = (j.get("eval_duration", 0) or 0) / 1e9
    return "LLM генерация", ed * 1000, f"{m}: {ec} токенов · {ec/ed if ed else 0:.1f} ток/с"


_BENCH_STEPS = [_bench_embed, _bench_rerank, _bench_search, _bench_qdrant, _bench_llm]


def stop_benchmark() -> dict:
    if not _bench_job["running"]:
        return {"ok": False, "msg": "бенчмарк не запущен"}
    _bench_job["cancel"] = True
    return {"ok": True, "msg": "остановка запрошена (после текущего шага)"}


def benchmark() -> dict:
    if _bench_job["running"]:
        return {"ok": False, "msg": "бенчмарк уже идёт"}
    logfile = "/tmp/rag_benchmark.log"

    def run():
        _bench_job.update(running=True, started=time.time(), finished=None, ok=None,
                          log="", logfile=logfile, results=[], cancel=False)
        results = []
        with open(logfile, "w", buffering=1, errors="ignore") as fp:
            fp.write("=== Бенчмарк производительности ===\n")
            stopped = False
            for step in _BENCH_STEPS:
                if _bench_job.get("cancel"):
                    fp.write("⏹ Остановлено пользователем.\n")
                    stopped = True
                    break
                try:
                    comp, ms, detail = step()
                except Exception as e:
                    comp, ms, detail = step.__name__, 0, f"ошибка: {e}"
                row = {"component": comp, "ms": round(ms), "detail": detail}
                results.append(row)
                _bench_job["results"] = list(results)
                fp.write(f"{comp}: {round(ms)} мс — {detail}\n")
                fp.flush()
            fp.write("\n" + ("Остановлено." if stopped else "Готово.") + "\n")
        _bench_job["results"] = results
        _bench_job["log"] = _tail(logfile)
        _bench_job["ok"] = not _bench_job.get("cancel")
        _bench_job["running"] = False
        _bench_job["finished"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True, "msg": "бенчмарк запущен"}


# ============================ самотесты компонентов ============================
def _t_settings():
    return True, (f"backend={settings.get('LLM_BACKEND')}, device={settings.get('DEVICE')}, "
                  f"model={settings.get('LLM_MODEL')}")


def _t_qdrant():
    coll = settings.get("QDRANT_COLLECTION")
    r = httpx.get(f"{settings.get('QDRANT_URL')}/collections/{coll}", timeout=6)
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}"
    n = (r.json().get("result", {}) or {}).get("points_count", 0)
    return True, f"коллекция «{coll}», чанков: {n}"


def _t_embedder():
    from retriever import _embedder
    v = _embedder().encode(["проверка эмбеддера"], normalize_embeddings=True)
    return len(v[0]) > 0, f"модель {settings.get('EMBED_MODEL')}, размерность {len(v[0])}"


def _t_reranker():
    from retriever import _reranker
    s = _reranker().compute_score([["вопрос", "ответ на вопрос"]], normalize=True)
    val = float(s[0] if isinstance(s, list) else s)
    return True, f"score={val:.3f}"


def _t_llm():
    b = settings.get("LLM_BACKEND")
    m = settings.get("LLM_MODEL")
    if b == "openai":
        r = httpx.post(f"{settings.get('LLM_BASE_URL')}/chat/completions",
                       headers={"Authorization": f"Bearer {settings.get('LLM_API_KEY')}"},
                       json={"model": m, "messages": [{"role": "user", "content": "Ответь словом: тест"}],
                             "max_tokens": 8, "temperature": 0}, timeout=60)
        txt = r.json()["choices"][0]["message"]["content"]
    else:
        r = httpx.post(f"{settings.get('OLLAMA_URL')}/api/generate",
                       json={"model": m, "prompt": "Ответь словом: тест", "stream": False,
                             "options": {"num_predict": 8}}, timeout=120)
        txt = r.json().get("response", "")
    return bool(txt.strip()), f"{b}/{m}: «{txt.strip()[:50]}»"


def _t_docs():
    p = Path(settings.get("DOCS_DIR")).expanduser()
    if not p.exists():
        return False, f"папка не найдена: {p}"
    n = sum(1 for f in p.rglob("*") if f.is_file())
    return True, f"{p}: файлов {n}"


def _t_db():
    import db
    return True, f"журнал: запросов {db.stats()['total']}"


def _t_graph():
    inst = _iu.find_spec("lightrag") is not None
    built = (ROOT / "graph_storage").exists()
    return (inst and built), f"установлен={inst}, граф построен={built}"


def _t_finetune():
    deps = _iu.find_spec("peft") is not None and _iu.find_spec("trl") is not None
    adapter = (ROOT / "finetune" / "adapter").exists()
    return (deps and adapter), f"зависимости={deps}, адаптер={adapter}"


# критичные компоненты (для общего вердикта); граф и дообучение — опциональны
_TESTS = [
    ("Настройки", _t_settings, True),
    ("Qdrant", _t_qdrant, True),
    ("Эмбеддер (bge-m3)", _t_embedder, True),
    ("Реранкер", _t_reranker, True),
    ("LLM (генерация)", _t_llm, True),
    ("Папка документов", _t_docs, True),
    ("Журнал (SQLite)", _t_db, True),
    ("LightRAG / граф", _t_graph, False),
    ("Дообучение (LoRA)", _t_finetune, False),
]


def self_test() -> dict:
    if _test_job["running"]:
        return {"ok": False, "msg": "тестирование уже идёт"}
    logfile = "/tmp/rag_selftest.log"

    def run():
        _test_job.update(running=True, started=time.time(), finished=None, ok=None,
                         log="", logfile=logfile, results=[])
        results = []
        with open(logfile, "w", buffering=1, errors="ignore") as fp:
            fp.write("=== Тестирование компонентов RAG ===\n")
            for name, fn, critical in _TESTS:
                fp.write(f"[ТЕСТ] {name} ...\n")
                fp.flush()
                try:
                    ok, detail = fn()
                except Exception as e:
                    ok, detail = False, str(e)[:300]
                results.append({"name": name, "ok": bool(ok), "detail": detail,
                                "critical": critical})
                fp.write(("  ✓ OK   " if ok else "  ✗ FAIL ") + f"{name}: {detail}\n")
                fp.flush()
                _test_job["results"] = list(results)
            crit = [r for r in results if r["critical"]]
            passed = sum(1 for r in crit if r["ok"])
            overall = passed == len(crit)
            fp.write(f"\nИТОГ: ключевых пройдено {passed}/{len(crit)}; "
                     f"всего {sum(1 for r in results if r['ok'])}/{len(results)}. "
                     f"Общий результат: {'УСПЕХ' if overall else 'ЕСТЬ ПРОБЛЕМЫ'}\n")
        _test_job["results"] = results
        _test_job["log"] = _tail(logfile)
        _test_job["ok"] = overall
        _test_job["running"] = False
        _test_job["finished"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True, "msg": "тестирование запущено"}


_WEB_SOURCES = ROOT / "web_sources.txt"


def _web_slug(url: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9._-]", "_", url)[:120] or "page"


def _web_list() -> list:
    """Список сохранённых сайтов: URL + файл + признак наличия в папке."""
    urls = []
    if _WEB_SOURCES.exists():
        urls = [u.strip() for u in _WEB_SOURCES.read_text(encoding="utf-8").splitlines() if u.strip()]
    webdir = Path(settings.get("DOCS_DIR")).expanduser() / "web"
    out = []
    for u in urls:
        rel = f"web/{_web_slug(u)}.html"
        out.append({"url": u, "source": rel, "indexed": (webdir / f"{_web_slug(u)}.html").exists()})
    return out


def get_web_urls() -> dict:
    sites = _web_list()
    return {"urls": [s["url"] for s in sites], "sites": sites,
            "log": _tail(_web_job["logfile"]) if _web_job.get("logfile") else _web_job.get("log", "")}


def delete_web(url: str) -> dict:
    """Удалить сайт из списка, его файл и чанки из базы знаний (Qdrant)."""
    url = (url or "").strip()
    if not url:
        return {"ok": False, "msg": "не указан URL"}
    # 1) убрать из списка
    sites = [s["url"] for s in _web_list() if s["url"] != url]
    _WEB_SOURCES.write_text("\n".join(sites), encoding="utf-8")
    # 2) удалить файл
    slug = _web_slug(url)
    f = Path(settings.get("DOCS_DIR")).expanduser() / "web" / f"{slug}.html"
    if f.exists():
        try:
            f.unlink()
        except Exception:
            pass
    # 3) удалить чанки из Qdrant по source
    src = f"web/{slug}.html"
    try:
        httpx.post(
            f"{settings.get('QDRANT_URL')}/collections/{settings.get('QDRANT_COLLECTION')}/points/delete",
            json={"filter": {"must": [{"key": "source", "match": {"value": src}}]}},
            timeout=30)
    except Exception as e:
        return {"ok": True, "msg": f"удалён из списка и папки; из Qdrant не удалён: {e}"}
    return {"ok": True, "msg": "сайт удалён из базы знаний"}


# расширения «страниц» (их обходим как HTML); всё остальное считаем файлами и скачиваем
_WEB_PAGE_EXT = {"", ".html", ".htm", ".php", ".asp", ".aspx", ".jsp", ".cfm", ".shtml"}


def _web_is_file(url: str) -> bool:
    from urllib.parse import urlparse
    return os.path.splitext(urlparse(url).path)[1].lower() not in _WEB_PAGE_EXT


class _Renderer:
    """Однократно запущенный headless-Chromium (Playwright) для JS-страниц."""

    def __init__(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._b = self._pw.chromium.launch(
            args=["--no-sandbox", "--disable-dev-shm-usage"])

    def render(self, url: str):
        try:
            pg = self._b.new_page(user_agent="Mozilla/5.0 (RAGBot)")
            try:
                pg.goto(url, wait_until="networkidle", timeout=45000)
            except Exception:
                pg.goto(url, wait_until="domcontentloaded", timeout=45000)
            html = pg.content()
            pg.close()
            return html
        except Exception as e:
            print(f"[web] render {url}: {e}")
            return None

    def close(self):
        for fn in (getattr(self._b, "close", None), getattr(self._pw, "stop", None)):
            try:
                if fn:
                    fn()
            except Exception:
                pass


def _web_fetch(url: str, renderer, log):
    """HTML страницы: через headless-браузер (если включён и доступен) или httpx."""
    if renderer is not None:
        html = renderer.render(url)
        if html:
            return html
    try:
        r = httpx.get(url, timeout=30, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0 (RAGBot)"})
        ct = r.headers.get("content-type", "")
        if ct and "html" not in ct.lower():
            return None
        return r.text
    except Exception as e:
        log(f"ERR {url}: {e}")
        return None


def _web_download(url: str, dest_dir, log):
    """Скачать файл по ссылке (любого типа) в dest_dir. Возвращает путь или None."""
    import re
    from urllib.parse import urlparse, unquote
    try:
        r = httpx.get(url, timeout=60, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0 (RAGBot)"})
        if r.status_code != 200:
            log(f"ERR файл {url}: HTTP {r.status_code}")
            return None
        name = unquote(os.path.basename(urlparse(url).path)) or _web_slug(url)
        name = re.sub(r"[^\w.\-]+", "_", name)[:150] or "file"
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = dest_dir / name
        i = 1
        while out.exists() and out.stat().st_size != len(r.content):
            out = dest_dir / f"{i}_{name}"
            i += 1
        out.write_bytes(r.content)
        log(f"ФАЙЛ {url} -> {out.name} ({len(r.content)} б)")
        return out
    except Exception as e:
        log(f"ERR файл {url}: {e}")
        return None


def _web_extract(html: str) -> str:
    """Извлечь основной текст страницы. trafilatura (если установлена) даёт чистый
    контент с таблицами; иначе — BeautifulSoup с предпочтением <main>/<article>."""
    try:
        import trafilatura
        txt = trafilatura.extract(html, include_tables=True, include_comments=False,
                                  favor_recall=True)
        if txt and len(txt.strip()) > 40:
            return txt.strip()
    except Exception:
        pass
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script", "style", "noscript", "header", "footer", "nav",
                       "aside", "form"]):
            t.decompose()
        main = soup.find("main") or soup.find("article") or soup.body or soup
        lines = [ln.strip() for ln in main.get_text(separator="\n").splitlines()]
        return "\n".join(ln for ln in lines if ln)
    except Exception:
        return ""


def _web_title(html: str, url: str) -> str:
    try:
        from bs4 import BeautifulSoup
        s = BeautifulSoup(html, "html.parser")
        if s.title and s.title.string:
            return s.title.string.strip()
    except Exception:
        pass
    return url


def _web_links(html: str, base: str, seed_netloc: str, same_domain: bool) -> list:
    """Все ссылки страницы (http/https, без mailto/tel/якорей). Без фильтра по типу —
    классификация на «страницы»/«файлы» делается в обходе."""
    from urllib.parse import urljoin, urlparse
    out = []
    try:
        from bs4 import BeautifulSoup
        s = BeautifulSoup(html, "html.parser")
        for a in s.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
                continue
            pr = urlparse(urljoin(base, href))
            if pr.scheme not in ("http", "https"):
                continue
            if same_domain and pr.netloc.replace("www.", "") != \
                    seed_netloc.replace("www.", ""):
                continue
            out.append(pr._replace(fragment="").geturl())
    except Exception:
        pass
    return out


def _web_crawl(seed: str, depth: int, max_pages: int, same_domain: bool, renderer, log):
    """Обойти сайт из стартовой страницы (BFS) до depth/max_pages. Возвращает
    (pages, files): pages — список (url, title, text); files — множество URL файлов
    (любого типа) для скачивания. log(msg) — журналирование."""
    from urllib.parse import urlparse
    seed_netloc = urlparse(seed).netloc
    seen, queue, pages, files = set(), [(seed, 0)], [], set()
    while queue and len(pages) < max_pages:
        url, d = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        if _web_is_file(url):          # сам адрес — файл (например, прямая ссылка на PDF)
            files.add(url)
            continue
        html = _web_fetch(url, renderer, log)
        if html is None:
            continue
        text = _web_extract(html)
        if text:
            pages.append((url, _web_title(html, url), text))
            log(f"OK  {url}  ({len(text)} симв.)")
        # ссылки: файлы собираем всегда, страницы — пока не достигли глубины
        for link in _web_links(html, url, seed_netloc, same_domain):
            if link in seen:
                continue
            if _web_is_file(link):
                files.add(link)
            elif d < depth and all(link != q[0] for q in queue):
                queue.append((link, d + 1))
    return pages, files


def ingest_web(urls: list) -> dict:
    """Скачать до 50 сайтов (с обходом ссылок), извлечь текст в DOCS_DIR/web и
    переиндексировать. Глубина/лимит/домен — настройки WEB_CRAWL_DEPTH/MAX_PAGES/SAME_DOMAIN."""
    import re
    urls = [u.strip() for u in (urls or []) if u.strip().startswith(("http://", "https://"))][:50]
    if not urls:
        return {"ok": False, "msg": "укажите хотя бы один URL (http/https), максимум 50"}
    if _web_job["running"]:
        return {"ok": False, "msg": "парсинг сайтов уже идёт"}
    _WEB_SOURCES.write_text("\n".join(urls), encoding="utf-8")
    webdir = Path(settings.get("DOCS_DIR")).expanduser() / "web"
    logfile = "/tmp/rag_web.log"
    _slug = _web_slug

    def run():
        _web_job.update(running=True, started=time.time(), finished=None, ok=None,
                        log="", summary="", logfile=logfile)
        ok = err = 0
        rc = -1
        try:
            import html as _html
            webdir.mkdir(parents=True, exist_ok=True)
            web_paths = []
            depth = max(0, int(settings.get("WEB_CRAWL_DEPTH") or 0))
            max_pages = max(1, int(settings.get("WEB_MAX_PAGES") or 1))
            max_files = max(0, int(settings.get("WEB_MAX_FILES") or 0))
            same_domain = bool(settings.get("WEB_SAME_DOMAIN"))
            filesdir = webdir / "files"
            with open(logfile, "w", buffering=1, errors="ignore") as fp:
                def _log(m):
                    fp.write(m + "\n")
                    fp.flush()

                # headless-браузер (Playwright) — один на весь прогон, если включён/доступен
                renderer = None
                if settings.get("WEB_JS_RENDER"):
                    try:
                        renderer = _Renderer()
                        _log("Headless-браузер (Playwright Chromium) активен")
                    except Exception as e:
                        _log(f"Headless-браузер недоступен ({e}); обычная загрузка. "
                             "Установка: pip install playwright && playwright install chromium")
                        renderer = None
                fp.write(f"Парсинг: глубина {depth}, до {max_pages} стр. и {max_files} "
                         f"файлов/сайт, "
                         f"{'тот же домен' if same_domain else 'любой домен'}\n")
                try:
                    for u in urls:
                        try:
                            pages, file_urls = _web_crawl(u, depth, max_pages,
                                                          same_domain, renderer, _log)
                            # скачиваем найденные файлы (любого типа), лимит на сайт
                            dl = 0
                            for furl in list(file_urls)[:max_files]:
                                p = _web_download(furl, filesdir, _log)
                                if p is not None:
                                    web_paths.append(p)
                                    dl += 1
                            # текст страниц — в один документ web/<slug>.html
                            if pages:
                                parts, total = [], 0
                                for (pu, pt, ptext) in pages:
                                    parts.append(
                                        "<h2>%s</h2>\n<p><small>%s</small></p>\n<pre>%s</pre>"
                                        % (_html.escape(pt or pu), _html.escape(pu),
                                           _html.escape(ptext)))
                                    total += len(ptext)
                                doc = ("<html><body><!-- source: %s -->\n<h1>%s</h1>\n"
                                       "%s</body></html>"
                                       % (_html.escape(u), _html.escape(pages[0][1] or u),
                                          "\n".join(parts)))
                                out = webdir / (_slug(u) + ".html")
                                out.write_text(doc, encoding="utf-8")
                                web_paths.append(out)
                            if pages or dl:
                                ok += 1
                                fp.write(f"ИТОГО {u}: страниц {len(pages)}, файлов {dl}\n")
                            else:
                                err += 1
                                fp.write(f"ERR {u}: ни текста, ни файлов (пустая страница "
                                         "или JS-сайт без headless-браузера)\n")
                        except Exception as e:
                            err += 1
                            fp.write(f"ERR {u}: {e}\n")
                        fp.flush()
                finally:
                    if renderer is not None:
                        renderer.close()
                # активен каталог PostgreSQL — кладём спарсенные страницы и в него,
                # чтобы индексация из БД их увидела (без папки)
                added = catalog_add_paths(web_paths)
                if added:
                    fp.write(f"В PostgreSQL добавлено страниц: {added}\n")
                fp.write(f"Скачано: {ok}, ошибок: {err}. Запускаю индексацию...\n")
                fp.flush()
                rc = subprocess.Popen([sys.executable, "-u", "ingest.py"], cwd=ROOT,
                                      stdout=fp, stderr=subprocess.STDOUT).wait(timeout=24 * 3600)
                fp.write(f"SUMMARY web_ok={ok} web_err={err} index_rc={rc}\n")
            _web_job["log"] = _tail(logfile)
            _web_job["summary"] = _extract_summary(_web_job["log"])
            _web_job["ok"] = (err == 0 and rc == 0)
        except Exception as e:
            _web_job["ok"] = False
            _web_job["log"] = (_tail(logfile) + "\n" + str(e)).strip()
        _web_job["running"] = False
        _web_job["finished"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True, "msg": f"парсинг {len(urls)} сайт(ов) запущен; затем — индексация"}


def _run_dep_job(label: str, cmd: list, timeout: int = 3600) -> dict:
    """Установочная команда в фоне (живой лог в «Состояние и операции»)."""
    return _bg(_dep_job, label, [cmd], "/tmp/rag_dep.log", timeout=timeout)


def install_lightrag() -> dict:
    """pip-установка LightRAG и зависимостей (без построения графа)."""
    return _run_dep_job("LightRAG", [sys.executable, "-m", "pip", "install", "-q", *_LIGHTRAG_DEPS])


def _docker_bin() -> str | None:
    """Найти docker даже если PATH урезан (launchd/systemd)."""
    d = shutil.which("docker")
    if d:
        return d
    for p in ("/usr/local/bin/docker", "/opt/homebrew/bin/docker", "/usr/bin/docker"):
        if os.path.exists(p):
            return p
    return None


def install_qdrant() -> dict:
    """Поднять контейнер Qdrant через docker compose."""
    docker = _docker_bin()
    if not docker:
        if os.path.exists("/Applications/Docker.app"):
            return {"ok": False, "msg": "Docker Desktop установлен, но не в PATH/не запущен. "
                    "Откройте приложение Docker и повторите."}
        return {"ok": False, "msg": "Docker не установлен. Mac: brew install --cask docker, затем "
                "откройте Docker Desktop. Linux: curl -fsSL https://get.docker.com | sh"}
    # проверка, что демон запущен
    try:
        info = subprocess.run([docker, "info"], capture_output=True, text=True, timeout=20)
        if info.returncode != 0:
            return {"ok": False, "msg": "Docker установлен, но демон не запущен — откройте Docker Desktop "
                    "(Mac) или запустите службу docker (Linux) и повторите."}
    except Exception as e:
        return {"ok": False, "msg": f"Docker недоступен: {e}"}
    compose = ROOT / "docker-compose.yml"
    if not compose.exists():
        compose = ROOT / "gpu_variant" / "docker-compose.gpu.yml"
    return _run_dep_job("Qdrant",
                        [docker, "compose", "-f", str(compose), "up", "-d", "qdrant"],
                        timeout=900)


def list_models() -> dict:
    """Список доступных моделей генерации из текущего бэкенда."""
    backend = settings.get("LLM_BACKEND")
    out = {"backend": backend, "current": settings.get("LLM_MODEL"), "models": []}
    try:
        if backend == "openai":
            r = httpx.get(f"{settings.get('LLM_BASE_URL')}/models",
                          headers={"Authorization": f"Bearer {settings.get('LLM_API_KEY')}"},
                          timeout=4)
            if r.status_code == 200:
                out["models"] = [m.get("id") for m in r.json().get("data", []) if m.get("id")]
        else:
            r = httpx.get(f"{settings.get('OLLAMA_URL')}/api/tags", timeout=4)
            if r.status_code == 200:
                out["models"] = [m.get("name") for m in r.json().get("models", []) if m.get("name")]
    except Exception as e:
        out["error"] = str(e)
    return out


# Курируемые каталоги моделей (полного API библиотеки у Ollama нет)
_OLLAMA_CATALOG = [
    # --- Qwen3.6 (новейшие, hybrid-thinking, 256K контекст) ---
    {"name": "qwen3.6:35b-a3b", "note": "MoE 35B/3B активных · ~24 ГБ · 256K"},
    {"name": "qwen3.6:35b-a3b-q4_K_M", "note": "~19–22 ГБ · квантованная MoE (по умолчанию ✅)"},
    {"name": "qwen3.6:27b", "note": "плотная 27B · ~17 ГБ (Q4) · 256K"},
    {"name": "qwen3.6:27b-bf16", "note": "~56 ГБ · полная точность"},
    # --- Qwen3 (гибридный reasoning) ---
    {"name": "qwen3:0.6b", "note": "~0.5 ГБ · самая лёгкая Qwen3"},
    {"name": "qwen3:1.7b", "note": "~1.4 ГБ · лёгкая"},
    {"name": "qwen3:4b", "note": "~2.6 ГБ"},
    {"name": "qwen3:8b", "note": "~5 ГБ · хороший баланс"},
    {"name": "qwen3:14b", "note": "~9 ГБ"},
    {"name": "qwen3:30b-a3b", "note": "~18 ГБ · MoE (быстрый при большом размере)"},
    {"name": "qwen3:32b", "note": "~20 ГБ · сильная, RU"},
    {"name": "qwen3:235b-a22b", "note": "~140 ГБ · топ MoE (нужен мощный сервер)"},
    # --- Qwen2.5 (сильный русский) ---
    {"name": "qwen2.5:3b-instruct", "note": "~2 ГБ · очень лёгкая"},
    {"name": "qwen2.5:7b-instruct", "note": "~4.7 ГБ · быстрый, базовый RU"},
    {"name": "qwen2.5:14b-instruct", "note": "~9 ГБ · хороший баланс"},
    {"name": "qwen2.5:32b-instruct-q4_K_M", "note": "~20 ГБ · сильный RU"},
    {"name": "qwen2.5:72b-instruct-q4_K_M", "note": "~42 ГБ · максимум качества"},
    # --- Llama 3.x ---
    {"name": "llama3.2:3b-instruct-q4_K_M", "note": "~2 ГБ · лёгкая"},
    {"name": "llama3.1:8b-instruct-q4_K_M", "note": "~4.9 ГБ"},
    {"name": "llama3.3:70b-instruct-q4_K_M", "note": "~42 ГБ · топ Llama"},
    # --- Многоязычные / RAG-ориентированные ---
    {"name": "aya-expanse:8b", "note": "~5 ГБ · многоязычная (Cohere), хороший RU"},
    {"name": "aya-expanse:32b", "note": "~18 ГБ · многоязычная, сильный RU"},
    {"name": "command-r7b", "note": "~5 ГБ · заточена под RAG/цитирование"},
    {"name": "command-r:35b", "note": "~20 ГБ · RAG, длинный контекст"},
    # --- Gemma 3 (140+ языков, мультимодальная) ---
    {"name": "gemma3:4b", "note": "~3 ГБ · многоязычная, есть зрение"},
    {"name": "gemma3:12b", "note": "~8 ГБ · многоязычная"},
    {"name": "gemma3:27b", "note": "~17 ГБ · сильная, 140+ языков"},
    # --- Gemma 2 / Mistral / Phi ---
    {"name": "gemma2:9b-instruct-q4_K_M", "note": "~5.8 ГБ"},
    {"name": "gemma2:27b-instruct-q4_K_M", "note": "~16 ГБ"},
    {"name": "mistral-small:24b", "note": "~14 ГБ · плотная, хороша для RAG"},
    {"name": "mistral-nemo:12b-instruct-2407-q4_K_M", "note": "~7 ГБ · 128k контекст"},
    {"name": "mixtral:8x7b-instruct-v0.1-q4_K_M", "note": "~26 ГБ · MoE"},
    {"name": "phi4:14b", "note": "~9 ГБ · сильная логика"},
    {"name": "phi3.5:3.8b-mini-instruct-q4_K_M", "note": "~2.2 ГБ · лёгкая"},
    # --- Reasoning (DeepSeek-R1) ---
    {"name": "deepseek-r1:7b", "note": "~4.7 ГБ · рассуждения"},
    {"name": "deepseek-r1:14b", "note": "~9 ГБ · рассуждения"},
    {"name": "deepseek-r1:32b", "note": "~20 ГБ · рассуждения"},
    # --- Эмбеддинги (для графа на Ollama) ---
    {"name": "bge-m3", "note": "эмбеддинги · многоязычные, 1024d (рекоменд.)"},
    {"name": "qwen3-embedding:0.6b", "note": "эмбеддинги · топ MTEB 2026, многоязычные"},
    {"name": "nomic-embed-text", "note": "эмбеддинги · лёгкие, многоязычные"},
    {"name": "mxbai-embed-large", "note": "эмбеддинги · качественные (англ.)"},
]
_VLLM_CATALOG = [
    # --- Qwen3.6 (новейшие, 256K контекст) ---
    {"name": "Qwen/Qwen3.6-35B-A3B", "note": "MoE 35B/3B · 24 ГБ VRAM (Q4) · 256K"},
    {"name": "Qwen/Qwen3.6-27B", "note": "плотная 27B · ~17 ГБ (Q4)"},
    {"name": "nvidia/Qwen3.6-35B-A3B-NVFP4", "note": "NVFP4-квант (NVIDIA), компактная"},
    # --- Qwen3 ---
    {"name": "Qwen/Qwen3-8B", "note": "24 ГБ VRAM · гибрид reasoning"},
    {"name": "Qwen/Qwen3-14B", "note": "32 ГБ VRAM"},
    {"name": "Qwen/Qwen3-32B", "note": "80 ГБ VRAM (или AWQ/TP=2)"},
    {"name": "Qwen/Qwen3-30B-A3B", "note": "48+ ГБ · MoE"},
    {"name": "Qwen/Qwen3-32B-AWQ", "note": "48 ГБ VRAM · квантованная"},
    # --- Qwen2.5 ---
    {"name": "Qwen/Qwen2.5-3B-Instruct-AWQ", "note": "16 ГБ VRAM"},
    {"name": "Qwen/Qwen2.5-7B-Instruct-AWQ", "note": "24 ГБ VRAM"},
    {"name": "Qwen/Qwen2.5-14B-Instruct-AWQ", "note": "24 ГБ VRAM"},
    {"name": "Qwen/Qwen2.5-32B-Instruct-AWQ", "note": "48 ГБ VRAM"},
    {"name": "Qwen/Qwen2.5-72B-Instruct-AWQ", "note": "80 ГБ VRAM / TP=2"},
    # --- Llama (AWQ от hugging-quants) ---
    {"name": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", "note": "24 ГБ VRAM"},
    {"name": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4", "note": "80 ГБ VRAM"},
    # --- Gemma 3 / Mistral ---
    {"name": "google/gemma-3-12b-it", "note": "32 ГБ · многоязычная, мультимодальная"},
    {"name": "google/gemma-3-27b-it", "note": "80 ГБ (или AWQ) · 140+ языков"},
    {"name": "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "note": "48 ГБ · плотная, RAG"},
    {"name": "casperhansen/mistral-nemo-instruct-2407-awq", "note": "24 ГБ · 128k контекст"},
    {"name": "casperhansen/mixtral-instruct-awq", "note": "48 ГБ · MoE"},
    {"name": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4", "note": "48 ГБ · альтернатива AWQ"},
]


def available_models() -> dict:
    """Каталог рекомендованных моделей под текущий бэкенд."""
    b = settings.get("LLM_BACKEND")
    return {"backend": b, "installable": b == "ollama",
            "catalog": _OLLAMA_CATALOG if b == "ollama" else _VLLM_CATALOG}


def vllm_models() -> dict:
    """Модели для vLLM: курируемый каталог + реально обслуживаемые сервером сейчас
    (через OpenAI-совместимый /v1/models). Доступно независимо от текущего бэкенда —
    чтобы можно было выбрать модель vLLM заранее."""
    served, err = [], None
    try:
        r = httpx.get(f"{settings.get('LLM_BASE_URL')}/models",
                      headers={"Authorization": f"Bearer {settings.get('LLM_API_KEY')}"},
                      timeout=4)
        if r.status_code == 200:
            served = [m.get("id") for m in r.json().get("data", []) if m.get("id")]
        else:
            err = f"vLLM вернул HTTP {r.status_code}"
    except Exception as e:
        err = str(e)
    return {"catalog": _VLLM_CATALOG, "served": served,
            "current": settings.get("VLLM_MODEL"),
            "base_url": settings.get("LLM_BASE_URL"), "error": err}


# Базовые (не-квантованные) fp16 HF-модели, пригодные для QLoRA-дообучения
_FINETUNE_CATALOG = [
    {"name": "Qwen/Qwen2.5-1.5B-Instruct", "note": "~8 ГБ VRAM · быстро, для проб"},
    {"name": "Qwen/Qwen2.5-3B-Instruct", "note": "~10 ГБ VRAM"},
    {"name": "Qwen/Qwen2.5-7B-Instruct", "note": "~16 ГБ VRAM · хороший баланс RU (рекоменд.)"},
    {"name": "Qwen/Qwen2.5-14B-Instruct", "note": "~24 ГБ VRAM"},
    {"name": "Qwen/Qwen2.5-32B-Instruct", "note": "~40 ГБ VRAM (QLoRA 4-bit)"},
    {"name": "Qwen/Qwen3-8B", "note": "~18 ГБ VRAM · гибрид reasoning"},
    {"name": "Qwen/Qwen3-14B", "note": "~24 ГБ VRAM"},
    {"name": "meta-llama/Llama-3.1-8B-Instruct", "note": "~16 ГБ VRAM · нужен доступ HF"},
    {"name": "meta-llama/Llama-3.2-3B-Instruct", "note": "~10 ГБ VRAM · лёгкая"},
    {"name": "google/gemma-2-9b-it", "note": "~18 ГБ VRAM · многоязычная"},
    {"name": "google/gemma-3-12b-it", "note": "~24 ГБ VRAM · 140+ языков"},
    {"name": "mistralai/Mistral-7B-Instruct-v0.3", "note": "~16 ГБ VRAM"},
    {"name": "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "note": "~40 ГБ VRAM"},
    {"name": "microsoft/Phi-4", "note": "~20 ГБ VRAM · сильная логика"},
]


def _strip_quant(m: str) -> str:
    for s in ("-AWQ", "-GPTQ", "-Int4", "-int4", "-GPTQ-Int4"):
        m = m.replace(s, "")
    return m


def finetune_models() -> dict:
    """Модели для дообучения (QLoRA): курируемый каталог fp16-баз + текущая
    выбранная база (FINETUNE_BASE или производная от VLLM_MODEL)."""
    explicit = (settings.get("FINETUNE_BASE") or "").strip()
    derived = _strip_quant(settings.get("VLLM_MODEL") or "Qwen/Qwen2.5-7B-Instruct")
    return {"catalog": _FINETUNE_CATALOG,
            "explicit": explicit, "derived": derived,
            "current": explicit or derived,
            "from_vllm": not explicit, "vllm_model": settings.get("VLLM_MODEL")}


def pull_model(name: str) -> dict:
    """Скачать новую модель в Ollama (фоном). Для vLLM — не применимо."""
    if settings.get("LLM_BACKEND") != "ollama":
        return {"ok": False, "msg": "Загрузка доступна только для Ollama. Для vLLM измените "
                "VLLM_MODEL и нажмите «Применить модель LLM»."}
    name = (name or "").strip()
    if not name:
        return {"ok": False, "msg": "укажите имя модели"}
    if _pull_job["running"]:
        return {"ok": False, "msg": "загрузка модели уже идёт"}

    def run():
        _pull_job.update(running=True, started=time.time(), finished=None,
                         ok=None, log="", model=name, status="запуск…",
                         percent=0, completed=0, total=0, speed=0)
        last_completed, last_t = 0, time.time()
        try:
            url = settings.get("OLLAMA_URL").rstrip("/") + "/api/pull"
            # Стримим прогресс из HTTP API Ollama (NDJSON: status/total/completed)
            with httpx.stream("POST", url, json={"model": name, "stream": True},
                              timeout=None) as r:
                if r.status_code != 200:
                    r.read()
                    raise RuntimeError(f"Ollama вернул HTTP {r.status_code}: {r.text[:300]}")
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        d = _json.loads(line)
                    except Exception:
                        continue
                    if d.get("error"):
                        _pull_job["ok"] = False
                        _pull_job["log"] = str(d["error"])
                        _pull_job["status"] = "ошибка"
                        break
                    if d.get("status"):
                        _pull_job["status"] = d["status"]
                    total, completed = d.get("total"), d.get("completed")
                    if total and completed is not None:
                        _pull_job["total"] = total
                        _pull_job["completed"] = completed
                        _pull_job["percent"] = round(completed * 100 / total, 1)
                        now = time.time()
                        if now - last_t >= 1.0:
                            _pull_job["speed"] = max(0, (completed - last_completed) / (now - last_t))
                            last_completed, last_t = completed, now
            if _pull_job["ok"] is None:
                _pull_job["ok"] = True
                _pull_job["status"] = "готово"
                _pull_job["percent"] = 100
        except Exception as e:
            # фолбэк: CLI-загрузка, если HTTP API недоступен
            _pull_job["status"] = "через CLI…"
            try:
                p = subprocess.run(["ollama", "pull", name], capture_output=True,
                                   text=True, timeout=6 * 3600)
                _pull_job["log"] = (p.stdout[-2000:] + "\n" + p.stderr[-2000:]).strip()
                _pull_job["ok"] = p.returncode == 0
                _pull_job["status"] = "готово" if p.returncode == 0 else "ошибка"
                if p.returncode == 0:
                    _pull_job["percent"] = 100
            except Exception as e2:
                _pull_job["ok"] = False
                _pull_job["log"] = f"{e}; CLI: {e2}"
                _pull_job["status"] = "ошибка"
        _pull_job["running"] = False
        _pull_job["finished"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True, "msg": f"загрузка модели {name} запущена"}


def reindex(reset: bool = False) -> dict:
    # python -u — небуферизованный вывод, чтобы лог индексации шёл вживую
    cmd = [sys.executable, "-u", "ingest.py"] + (["--reset"] if reset else [])
    r = _bg(_job, "Индексация", [cmd], "/tmp/rag_index.log")
    if r.get("ok"):
        r["msg"] = "индексация запущена"
    return r


def apply_llm() -> dict:
    """Перезапуск vLLM с текущей моделью из настроек (только GPU-вариант)."""
    script = ROOT / "gpu_variant" / "apply_llm.sh"
    env_file = ROOT / "gpu_variant" / ".env"
    if not script.exists():
        return {"ok": False, "msg": "apply_llm.sh не найден — это операция только для GPU-варианта"}
    _update_env(env_file, {
        "VLLM_MODEL": settings.get("VLLM_MODEL"),
        "VLLM_MAX_LEN": settings.get("VLLM_MAX_LEN"),
        "VLLM_TP": settings.get("VLLM_TP"),
        "LLM_MODEL": settings.get("LLM_MODEL"),
    })
    try:
        p = subprocess.run(["bash", str(script)], cwd=ROOT / "gpu_variant",
                           capture_output=True, text=True, timeout=1800)
        return {"ok": p.returncode == 0,
                "msg": (p.stdout + p.stderr)[-1500:].strip() or "vLLM перезапускается"}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


def build_graph() -> dict:
    """Установить LightRAG и построить граф знаний из документов (в фоне, живой лог)."""
    cmds = [[sys.executable, "-m", "pip", "install", "-q", *_LIGHTRAG_DEPS],
            [sys.executable, "-u", "-m", "graph_rag", "ingest"]]
    r = _bg(_graph_job, "Граф", cmds, "/tmp/rag_graph.log")
    if r.get("ok"):
        r["msg"] = "построение графа запущено (может занять часы)"
    return r


def finetune() -> dict:
    """Запустить пайплайн дообучения (датасет + LoRA) в фоне (живой лог)."""
    script = ROOT / "finetune" / "run_pipeline.sh"
    if not script.exists():
        return {"ok": False, "msg": "run_pipeline.sh не найден"}
    r = _bg(_ft_job, "Дообучение", [["bash", str(script)]], "/tmp/rag_finetune.log")
    if r.get("ok"):
        r["msg"] = "дообучение запущено (может занять часы)"
    return r


def apply_finetuned() -> dict:
    """Перезапустить vLLM с LoRA-адаптером (GPU)."""
    script = ROOT / "gpu_variant" / "apply_finetuned.sh"
    env_file = ROOT / "gpu_variant" / ".env"
    if not script.exists():
        return {"ok": False, "msg": "apply_finetuned.sh не найден — операция только для GPU-варианта"}
    if not (ROOT / "finetune" / "adapter").exists():
        return {"ok": False, "msg": "адаптер не найден — сначала запустите дообучение"}
    _update_env(env_file, {
        "VLLM_MODEL": settings.get("VLLM_MODEL"),
        "VLLM_MAX_LEN": settings.get("VLLM_MAX_LEN"),
        "VLLM_TP": settings.get("VLLM_TP"),
        "FINETUNED_MODEL": settings.get("FINETUNED_MODEL"),
    })
    try:
        p = subprocess.run(["bash", str(script)], cwd=ROOT / "gpu_variant",
                           capture_output=True, text=True, timeout=1800)
        return {"ok": p.returncode == 0,
                "msg": (p.stdout + p.stderr)[-1500:].strip() or "vLLM перезапускается с адаптером"}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


def reset(targets: list) -> dict:
    """Сброс выбранных данных. targets: index|graph|adapter|logs|settings|all."""
    targets = set(targets or [])
    if "all" in targets:
        targets |= {"index", "graph", "adapter", "logs", "settings"}
    done, errors = [], []

    if "index" in targets:
        base, coll = settings.get("QDRANT_URL"), settings.get("QDRANT_COLLECTION")
        try:
            httpx.delete(f"{base}/collections/{coll}", timeout=15)
            # пересоздаём пустую коллекцию, чтобы чат не падал до переиндексации
            httpx.put(f"{base}/collections/{coll}", timeout=15,
                      json={"vectors": {"size": 1024, "distance": "Cosine"}})
            # сбрасываем накопленное время обработки — оно относится к старому индексу
            _INGEST_STATS.unlink(missing_ok=True)
            try:
                import cache
                cache.bump("index")   # сброс кэша поиска/ответов
            except Exception:
                pass
            done.append("индекс Qdrant")
        except Exception as e:
            errors.append(f"индекс: {e}")

    if "graph" in targets:
        shutil.rmtree(ROOT / "graph_storage", ignore_errors=True)
        done.append("граф")

    if "adapter" in targets:
        shutil.rmtree(ROOT / "finetune" / "adapter", ignore_errors=True)
        shutil.rmtree(ROOT / "finetune" / "data", ignore_errors=True)
        done.append("адаптер и датасет")

    if "logs" in targets:
        try:
            db.clear()
            done.append("журнал")
        except Exception as e:
            errors.append(f"журнал: {e}")

    if "settings" in targets:
        settings.reset()
        done.append("настройки")

    return {"ok": not errors, "done": done, "errors": errors}


def reinstall_env() -> dict:
    """Переустановка окружения/зависимостей (фоновый detached-процесс reinstall.sh)."""
    script = ROOT / "reinstall.sh"
    if not script.exists():
        return {"ok": False, "msg": "reinstall.sh не найден"}
    try:
        logf = open("/tmp/rag_reinstall.log", "ab")
        subprocess.Popen(["bash", str(script)], cwd=ROOT,
                         stdout=logf, stderr=subprocess.STDOUT,
                         start_new_session=True)
        return {"ok": True, "msg": "переустановка окружения запущена; сервис перезапустится. "
                "Лог: /tmp/rag_reinstall.log"}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


_upd_job = {"running": False, "started": None, "finished": None, "ok": None, "log": ""}


def _git(*args):
    p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=120)
    return p.stdout.strip()


def check_updates() -> dict:
    """Сравнить локальную версию с origin (git fetch)."""
    try:
        if not (ROOT / ".git").exists():
            return {"ok": False, "msg": "это не git-репозиторий (обновление через git недоступно)"}
        subprocess.run(["git", "fetch", "--all", "-q"], cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
        branch = _git("rev-parse", "--abbrev-ref", "HEAD") or "main"
        local = _git("rev-parse", "--short", "HEAD")
        latest = _git("rev-parse", "--short", f"origin/{branch}")
        behind = _git("rev-list", "--count", f"HEAD..origin/{branch}")
        n = int(behind or 0)
        changes = _git("log", "--oneline", f"HEAD..origin/{branch}")
        return {"ok": True, "branch": branch, "current": local, "latest": latest,
                "behind": n, "up_to_date": n == 0, "changes": changes[:2000]}
    except FileNotFoundError:
        return {"ok": False, "msg": "git не установлен"}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


def update_app() -> dict:
    """git pull + зависимости в фоне, затем самоперезапуск (без sudo).
    Подхват новой версии — через systemd Restart=always / launchd KeepAlive."""
    if not (ROOT / ".git").exists():
        return {"ok": False, "msg": "это не git-репозиторий"}
    if _upd_job["running"]:
        return {"ok": False, "msg": "обновление уже идёт"}

    def run():
        _upd_job.update(running=True, started=time.time(), finished=None, ok=None, log="")
        out = []
        try:
            branch = _git("rev-parse", "--abbrev-ref", "HEAD") or "main"
            for cmd in (["git", "fetch", "--all", "-q"],
                        ["git", "reset", "--hard", f"origin/{branch}"]):
                p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
                out.append((p.stdout + p.stderr).strip())
                if p.returncode != 0:
                    raise RuntimeError(" ".join(cmd) + " → " + (p.stderr[-300:] or p.stdout[-300:]))
            req = "gpu_variant/requirements-gpu.txt" if shutil.which("nvidia-smi") else "requirements.txt"
            p = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", req],
                               cwd=ROOT, capture_output=True, text=True, timeout=3600)
            out.append((p.stdout[-800:] + p.stderr[-800:]).strip())
            _upd_job["ok"] = p.returncode == 0
            out.append("Готово, перезапуск сервиса...")
        except Exception as e:
            _upd_job["ok"] = False
            out.append(str(e))
        _upd_job["log"] = "\n".join(x for x in out if x)[-4000:]
        _upd_job["running"] = False
        _upd_job["finished"] = time.time()
        if _upd_job["ok"]:
            time.sleep(1)
            os._exit(0)  # супервизор (systemd/launchd) поднимет с новой версией

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True, "msg": "обновление запущено; сервис перезапустится автоматически"}


_CALIB_BACKUP = ROOT / "calib_backup.json"
_CALIB_KEYS = ("MIN_SCORE", "TOP_K_RETRIEVE", "TOP_K_RERANK", "TEMPERATURE",
               "AUTO_FILTER", "SMART_FILTER")


def recommend() -> dict:
    """Анализ оценок ответов → рекомендации по настройкам (эвристики)."""
    import db
    a = db.rating_analysis()
    cur = {k: settings.get(k) for k in _CALIB_KEYS}
    rated = (a["good_n"] or 0) + (a["bad_n"] or 0)
    if rated < 5:
        return {"ok": True, "enough": False, "analysis": a, "current": cur, "changes": {},
                "reasons": [], "msg": "Недостаточно оценок для анализа (нужно ≥ 5)."}
    changes, reasons = {}, []
    bad_n = a["bad_n"] or 1

    # 1) часто «плохо» из-за отказа «не знаю» → ослабить порог, расширить выборку
    if a["bad_no_answer"] >= max(2, 0.3 * bad_n):
        ms = round(max(0.15, cur["MIN_SCORE"] - 0.07), 2)
        if ms < cur["MIN_SCORE"]:
            changes["MIN_SCORE"] = ms
            reasons.append(f"Часто «не знаю» при плохих оценках → снизить MIN_SCORE до {ms}.")
        if cur["TOP_K_RETRIEVE"] < 30:
            changes["TOP_K_RETRIEVE"] = min(30, cur["TOP_K_RETRIEVE"] + 10)
            reasons.append(f"Увеличить TOP_K_RETRIEVE до {changes['TOP_K_RETRIEVE']}.")
        if cur["TOP_K_RERANK"] < 8:
            changes["TOP_K_RERANK"] = min(8, cur["TOP_K_RERANK"] + 2)
            reasons.append(f"Увеличить TOP_K_RERANK до {changes['TOP_K_RERANK']}.")

    # 2) часто «плохо» при наличии ответа → точнее/строже
    if a["bad_answered"] >= max(2, 0.5 * bad_n):
        if cur["TEMPERATURE"] > 0.1:
            changes["TEMPERATURE"] = 0.1
            reasons.append("Снизить TEMPERATURE до 0.1 (точнее ответы).")
        if not cur["SMART_FILTER"]:
            changes["SMART_FILTER"] = True
            reasons.append("Включить умные фильтры (SMART_FILTER).")
        if a["bad_avg_score"] is not None and a["good_avg_score"] is not None \
                and a["bad_avg_score"] < a["good_avg_score"]:
            ms = round(min(0.6, (a["bad_avg_score"] + a["good_avg_score"]) / 2), 2)
            if ms > cur["MIN_SCORE"] and "MIN_SCORE" not in changes:
                changes["MIN_SCORE"] = ms
                reasons.append(f"Поднять MIN_SCORE до {ms} (плохие ответы имеют низкую релевантность).")

    return {"ok": True, "enough": True, "analysis": a, "current": cur,
            "changes": changes, "reasons": reasons,
            "msg": "Изменений не требуется." if not changes else f"Рекомендовано изменений: {len(changes)}."}


def apply_recommendations() -> dict:
    rec = recommend()
    ch = rec.get("changes") or {}
    if not ch:
        return {"ok": True, "msg": rec.get("msg", "нет изменений")}
    backup = {k: settings.get(k) for k in ch}
    try:
        _CALIB_BACKUP.write_text(_json.dumps(backup, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    settings.update(ch)
    return {"ok": True, "msg": f"применено изменений: {len(ch)}", "changes": ch}


def rollback_calibration() -> dict:
    if not _CALIB_BACKUP.exists():
        return {"ok": False, "msg": "нет сохранённого состояния для отката"}
    try:
        backup = _json.loads(_CALIB_BACKUP.read_text(encoding="utf-8"))
        settings.update(backup)
        _CALIB_BACKUP.unlink()
        return {"ok": True, "msg": f"откат выполнен ({len(backup)} парам.)"}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


def reinstall_full(kind: str) -> dict:
    """Полная переустановка с нуля (destructive). kind: server (GPU) | mac.
    Запускает соответствующий скрипт detached с CONFIRM=yes."""
    scripts = {"server": ROOT / "reinstall_server.sh",
               "mac": ROOT / "mac_variant" / "reinstall_mac.sh"}
    sc = scripts.get(kind)
    if not sc or not sc.exists():
        return {"ok": False, "msg": f"скрипт переустановки '{kind}' не найден"}
    try:
        logf = open("/tmp/rag_reinstall.log", "ab")
        subprocess.Popen(["bash", str(sc)], cwd=ROOT,
                         env={**os.environ, "CONFIRM": "yes"},
                         stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        note = ("полная переустановка запущена; сервис будет недоступен во время "
                "процесса. Лог: /tmp/rag_reinstall.log")
        if kind == "server":
            note += " (для GPU нужны права root — запускайте сервис от пользователя с sudo)"
        return {"ok": True, "msg": note}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


def restart() -> dict:
    """Завершить процесс — systemd (Restart=always) поднимет его заново."""
    def killer():
        time.sleep(1)
        os._exit(0)
    threading.Thread(target=killer, daemon=True).start()
    return {"ok": True, "msg": "перезапуск сервиса через ~1 сек..."}


def _qcount(base: str, coll: str, flt: dict) -> int:
    try:
        r = httpx.post(f"{base}/collections/{coll}/points/count", timeout=4,
                       json={"filter": flt, "exact": True})
        if r.status_code == 200:
            return r.json().get("result", {}).get("count", 0)
    except Exception:
        pass
    return 0


def _extract_summary(text: str) -> str:
    """Достаёт машиночитаемую строку 'SUMMARY ...' из вывода задачи."""
    for line in reversed((text or "").splitlines()):
        if line.startswith("SUMMARY "):
            return line[len("SUMMARY "):].strip()
        if line.startswith("FATAL:"):
            return line.strip()
    return ""


def _dir_size_mb(path: Path) -> float:
    try:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return round(total / 1e6, 1)
    except Exception:
        return 0.0


def system_info() -> dict:
    """Сводка по компонентам с коротким кэшем (Redis, 3 с): дашборд опрашивает часто,
    кэш снижает нагрузку на Qdrant и повторные проверки внешних БД/Redis."""
    try:
        import cache
        return cache.get_or_set("system_info", 3, _system_info_raw, ns="live")
    except Exception:
        return _system_info_raw()


def _system_info_raw() -> dict:
    """Полная сводка по компонентам: Qdrant, граф (LightRAG), дообучение, hybrid+."""
    coll = settings.get("QDRANT_COLLECTION")
    qbase = settings.get("QDRANT_URL")

    # ---- Qdrant ----
    # «online» = сервер Qdrant доступен (проверяем по списку коллекций), отдельно —
    # существует ли наша коллекция. На свежей установке коллекции ещё нет (не было
    # индексации) — это НЕ значит, что Qdrant недоступен.
    qd: dict = {"online": False, "collection_exists": False}
    try:
        ping = httpx.get(f"{qbase}/collections", timeout=4)
        if ping.status_code == 200:
            qd["online"] = True
            qd["collection"] = coll
            r = httpx.get(f"{qbase}/collections/{coll}", timeout=4)
            if r.status_code == 200:
                res = r.json().get("result", {}) or {}
                params = (res.get("config", {}) or {}).get("params", {}) or {}
                vec = params.get("vectors", {}) or {}
                qd.update({
                    "collection_exists": True,
                    "points": res.get("points_count", 0),
                    "segments": res.get("segments_count", 0),
                    "status": res.get("status"),
                    "vector_size": vec.get("size"),
                    "distance": vec.get("distance"),
                    "payload_fields": sorted((res.get("payload_schema") or {}).keys()),
                    "by_category": {
                        cat: _qcount(qbase, coll,
                                     {"must": [{"key": "doc_category",
                                                "match": {"value": cat}}]})
                        for cat in ("price", "presentation", "training", "document")
                    },
                    "with_product": _qcount(qbase, coll,
                                            {"must_not": [{"is_empty": {"key": "product"}}]}),
                })
            else:
                qd["note"] = "коллекция ещё не создана — выполните «Переиндексировать»"
        else:
            qd["error"] = f"Qdrant вернул HTTP {ping.status_code}"
    except Exception as e:
        qd = {"online": False, "collection_exists": False, "error": str(e)}

    # ---- LightRAG / граф ----
    gdir = ROOT / "graph_storage"
    graph: dict = {
        "ready": gdir.exists(),
        "installed": _iu.find_spec("lightrag") is not None,
        "engine": settings.get("ENGINE"),
        "mode": settings.get("GRAPH_MODE"),
        "job": dict(_graph_job),
    }
    if gdir.exists():
        def _jlen(name, key=None):
            f = gdir / name
            if not f.exists():
                return None
            try:
                d = _json.loads(f.read_text(encoding="utf-8"))
                v = d.get(key) if key else d
                return len(v) if hasattr(v, "__len__") else None
            except Exception:
                return None
        graph["entities"] = _jlen("vdb_entities.json", "data")
        graph["relations"] = _jlen("vdb_relationships.json", "data")
        graph["chunks"] = _jlen("kv_store_text_chunks.json")
        graph["docs"] = _jlen("kv_store_full_docs.json")
        gml = gdir / "graph_chunk_entity_relation.graphml"
        if gml.exists() and graph.get("entities") is None:
            t = gml.read_text(errors="ignore")
            graph["entities"] = t.count("<node ")
            graph["relations"] = t.count("<edge ")
        graph["size_mb"] = _dir_size_mb(gdir)

    # ---- Дообучение ----
    adapter = ROOT / "finetune" / "adapter"
    ds = ROOT / "finetune" / "data" / "train.jsonl"
    ft: dict = {
        "adapter_ready": adapter.exists(),
        "use_finetuned": bool(settings.get("USE_FINETUNED")),
        "finetuned_model": settings.get("FINETUNED_MODEL"),
        "base_model": settings.get("VLLM_MODEL"),
        "deps_installed": (_iu.find_spec("peft") is not None
                           and _iu.find_spec("trl") is not None),
        "job": dict(_ft_job),
    }
    if ds.exists():
        try:
            ft["dataset_pairs"] = sum(1 for _ in ds.open(encoding="utf-8"))
        except Exception:
            ft["dataset_pairs"] = None
    if adapter.exists():
        ft["adapter_config"] = (adapter / "adapter_config.json").exists()
        ft["adapter_size_mb"] = _dir_size_mb(adapter)

    # ---- Hybrid+ ----
    hybrid = {
        "mode": settings.current_mode(),
        "LLM_METADATA": bool(settings.get("LLM_METADATA")),
        "SMART_FILTER": bool(settings.get("SMART_FILTER")),
        "GRAPH_RAG": bool(settings.get("GRAPH_RAG")),
        "AUTO_FILTER": bool(settings.get("AUTO_FILTER")),
        "GRAPH_MODE": settings.get("GRAPH_MODE"),
    }

    # ---- База данных и кэш ----
    try:
        database = db.system_stats()
    except Exception as e:
        database = {"active": "sqlite", "backends": {}, "error": str(e)}
    try:
        import cache
        cache_info = cache.status()
    except Exception as e:
        cache_info = {"enabled": False, "error": str(e)}

    return {"qdrant": qd, "graph": graph, "finetune": ft,
            "hybrid": hybrid, "usage": db.engine_usage(),
            "ingest": _ingest_summary(),
            "database": database, "cache": cache_info}


def _num(s):
    try:
        s = str(s).strip()
        return float(s) if "." in s else int(s)
    except Exception:
        return None


def _gpu_info() -> dict:
    """Данные о GPU: NVIDIA (nvidia-smi), AMD (rocm-smi) или Apple Silicon."""
    import platform
    g: dict = {"vendor": "none", "devices": []}

    # NVIDIA
    if shutil.which("nvidia-smi"):
        try:
            q = ("index,name,utilization.gpu,utilization.memory,memory.used,"
                 "memory.total,temperature.gpu,power.draw,power.limit,fan.speed")
            r = subprocess.run(["nvidia-smi", f"--query-gpu={q}",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=8)
            if r.returncode == 0 and r.stdout.strip():
                g["vendor"] = "nvidia"
                for line in r.stdout.strip().splitlines():
                    p = [x.strip() for x in line.split(",")]
                    if len(p) >= 9:
                        g["devices"].append({
                            "index": p[0], "name": p[1], "util": _num(p[2]),
                            "mem_util": _num(p[3]), "mem_used": _num(p[4]),
                            "mem_total": _num(p[5]), "temp": _num(p[6]),
                            "power": _num(p[7]), "power_limit": _num(p[8]),
                            "fan": _num(p[9]) if len(p) > 9 else None,
                        })
                return g
        except Exception as e:
            g["error"] = str(e)

    # AMD ROCm
    if shutil.which("rocm-smi"):
        g["vendor"] = "amd"
        g["devices"].append({"name": "AMD GPU (rocm-smi доступен)"})
        return g

    # Apple Silicon — единая память, live-загрузку GPU без sudo не получить
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        g["vendor"] = "apple"
        chip = "Apple Silicon"
        try:
            r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True, timeout=4)
            if r.stdout.strip():
                chip = r.stdout.strip()
        except Exception:
            pass
        g["devices"].append({"name": f"{chip} — встроенный GPU (Metal), единая память"})
    return g


def server_load() -> dict:
    """Подробная текущая загрузка хоста: CPU, память, диски, GPU, сеть, аптайм."""
    import platform
    out: dict = {"ts": time.time(), "platform": {
        "system": platform.system(), "release": platform.release(),
        "machine": platform.machine(), "python": platform.python_version(),
        "hostname": platform.node(),
    }}
    try:
        import psutil
    except Exception:
        psutil = None
    out["psutil"] = psutil is not None

    if psutil:
        try:
            freq = psutil.cpu_freq()
        except Exception:
            freq = None
        loadavg = None
        try:
            loadavg = [round(x, 2) for x in psutil.getloadavg()]
        except Exception:
            pass
        out["cpu"] = {
            "percent": psutil.cpu_percent(interval=0.3),
            "per_core": psutil.cpu_percent(interval=0.0, percpu=True),
            "cores_logical": psutil.cpu_count(),
            "cores_physical": psutil.cpu_count(logical=False),
            "freq_mhz": round(freq.current) if freq and freq.current else None,
            "loadavg": loadavg,
        }
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        out["memory"] = {
            "total": vm.total, "used": vm.used, "available": vm.available,
            "percent": vm.percent, "swap_total": sm.total,
            "swap_used": sm.used, "swap_percent": sm.percent,
        }
        disks = []
        for part in psutil.disk_partitions(all=False):
            try:
                u = psutil.disk_usage(part.mountpoint)
            except Exception:
                continue
            disks.append({"device": part.device, "mount": part.mountpoint,
                          "fstype": part.fstype, "total": u.total, "used": u.used,
                          "free": u.free, "percent": u.percent})
        out["disks"] = disks
        try:
            io = psutil.disk_io_counters()
            out["disk_io"] = {"read_bytes": io.read_bytes, "write_bytes": io.write_bytes}
        except Exception:
            pass
        try:
            nio = psutil.net_io_counters()
            out["net"] = {"sent": nio.bytes_sent, "recv": nio.bytes_recv}
        except Exception:
            pass
        try:
            out["uptime_sec"] = time.time() - psutil.boot_time()
        except Exception:
            pass
        procs = []
        for p in psutil.process_iter(["pid", "name", "memory_info", "cpu_percent"]):
            try:
                mi = p.info.get("memory_info")
                procs.append({"pid": p.info["pid"], "name": p.info.get("name") or "",
                              "rss": mi.rss if mi else 0,
                              "cpu": p.info.get("cpu_percent") or 0})
            except Exception:
                continue
        procs.sort(key=lambda x: x["rss"], reverse=True)
        out["top_processes"] = procs[:8]
    else:
        # фолбэк без psutil (минимальный набор)
        out["note"] = "Установите psutil для подробных метрик: pip install psutil"
        out["cpu"] = {"cores_logical": os.cpu_count()}
        try:
            out["cpu"]["loadavg"] = [round(x, 2) for x in os.getloadavg()]
        except Exception:
            pass
        try:
            mem = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                k, _, v = line.partition(":")
                mem[k] = int(v.strip().split()[0]) * 1024
            total, avail = mem.get("MemTotal", 0), mem.get("MemAvailable", 0)
            out["memory"] = {
                "total": total, "available": avail, "used": total - avail,
                "percent": round((total - avail) * 100 / total, 1) if total else None,
                "swap_total": mem.get("SwapTotal", 0),
                "swap_used": mem.get("SwapTotal", 0) - mem.get("SwapFree", 0),
            }
        except Exception:
            pass
        try:
            du = shutil.disk_usage("/")
            out["disks"] = [{"mount": "/", "total": du.total, "used": du.used,
                             "free": du.free,
                             "percent": round(du.used * 100 / du.total, 1)}]
        except Exception:
            pass

    out["gpu"] = _gpu_info()
    return out


def component_analytics() -> dict:
    """Расширенная аналитика по компонентам: Qdrant, граф (LightRAG), дообучение."""
    coll = settings.get("QDRANT_COLLECTION")
    qbase = settings.get("QDRANT_URL")

    # ---- Qdrant: по категориям, типам файлов, покрытию метаданными ----
    # online = сервер доступен; коллекции может не быть (свежая установка).
    qd: dict = {"online": False}
    try:
        ping = httpx.get(f"{qbase}/collections", timeout=4)
        if ping.status_code != 200:
            raise RuntimeError(f"HTTP {ping.status_code}")
        r = httpx.get(f"{qbase}/collections/{coll}", timeout=4)
        if r.status_code == 200:
            res = r.json().get("result", {}) or {}
            qd = {"online": True, "points": res.get("points_count", 0),
                  "segments": res.get("segments_count", 0)}
            qd["by_category"] = {
                c: _qcount(qbase, coll, {"must": [{"key": "doc_category", "match": {"value": c}}]})
                for c in ("price", "presentation", "training", "document")}
            byf = {}
            for ft in ("pdf", "docx", "pptx", "xlsx", "xls", "csv", "txt", "md",
                       "html", "htm", "mp3", "wav", "m4a", "aac", "mp4", "mov", "mkv", "webm"):
                n = _qcount(qbase, coll, {"must": [{"key": "ftype", "match": {"value": ft}}]})
                if n:
                    byf[ft] = n
            qd["by_ftype"] = byf
            qd["meta"] = {
                "product": _qcount(qbase, coll, {"must_not": [{"is_empty": {"key": "product"}}]}),
                "topic": _qcount(qbase, coll, {"must_not": [{"is_empty": {"key": "topic"}}]}),
                "doc_type": _qcount(qbase, coll, {"must_not": [{"is_empty": {"key": "doc_type"}}]}),
            }
    except Exception as e:
        qd = {"online": False, "error": str(e)}

    # ---- Граф (LightRAG) ----
    gdir = ROOT / "graph_storage"
    graph: dict = {"ready": gdir.exists()}
    if gdir.exists():
        def _jlen(name, key=None):
            f = gdir / name
            if not f.exists():
                return None
            try:
                d = _json.loads(f.read_text(encoding="utf-8"))
                v = d.get(key) if key else d
                return len(v) if hasattr(v, "__len__") else None
            except Exception:
                return None
        ent = _jlen("vdb_entities.json", "data")
        rel = _jlen("vdb_relationships.json", "data")
        ch = _jlen("kv_store_text_chunks.json")
        dc = _jlen("kv_store_full_docs.json")
        gml = gdir / "graph_chunk_entity_relation.graphml"
        if ent is None and gml.exists():
            t = gml.read_text(errors="ignore")
            ent, rel = t.count("<node "), t.count("<edge ")
        graph.update(entities=ent, relations=rel, chunks=ch, docs=dc,
                     size_mb=_dir_size_mb(gdir))
        if ent:
            graph["rel_per_entity"] = round((rel or 0) / ent, 2)

    # ---- Дообучение: датасет и параметры LoRA ----
    ft: dict = {"adapter_ready": (ROOT / "finetune" / "adapter").exists()}
    ds = ROOT / "finetune" / "data" / "train.jsonl"
    if ds.exists():
        pairs = ql = al = 0
        hist = {"<100": 0, "100–300": 0, "300–600": 0, ">600": 0}
        try:
            with ds.open(encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 5000:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = _json.loads(line)
                    except Exception:
                        continue
                    msgs = rec.get("messages", [])
                    q = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
                    a = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
                    pairs += 1
                    ql += len(q)
                    al += len(a)
                    L = len(a)
                    if L < 100:
                        hist["<100"] += 1
                    elif L < 300:
                        hist["100–300"] += 1
                    elif L < 600:
                        hist["300–600"] += 1
                    else:
                        hist[">600"] += 1
        except Exception:
            pass
        ft["dataset"] = {"pairs": pairs, "avg_q": round(ql / pairs) if pairs else 0,
                         "avg_a": round(al / pairs) if pairs else 0, "ans_hist": hist}
    acfg = ROOT / "finetune" / "adapter" / "adapter_config.json"
    if acfg.exists():
        try:
            c = _json.loads(acfg.read_text(encoding="utf-8"))
            ft["lora"] = {"r": c.get("r"), "alpha": c.get("lora_alpha"),
                          "dropout": c.get("lora_dropout"),
                          "targets": len(c.get("target_modules") or [])}
        except Exception:
            pass
        ft["adapter_size_mb"] = _dir_size_mb(ROOT / "finetune" / "adapter")

    # ---- тайминги: средние по этапам, длительности задач, последний бенчмарк ----
    st = db.stats()
    def _dur(j):
        if j.get("started") and j.get("finished"):
            return round(j["finished"] - j["started"], 1)
        return None
    timings = {
        "stages": {"retrieve": st.get("avg_retrieve_ms", 0),
                   "gen": st.get("avg_gen_ms", 0),
                   "total": st.get("avg_latency_ms", 0)},
        "jobs": [
            {"name": "Индексация", "sec": _dur(_job)},
            {"name": "Граф", "sec": _dur(_graph_job)},
            {"name": "Дообучение", "sec": _dur(_ft_job)},
            {"name": "Парсинг сайтов", "sec": _dur(_web_job)},
        ],
        "benchmark": [{"component": r["component"], "ms": r["ms"]} for r in _bench_job.get("results", [])],
        "ingest": _ingest_summary(),
    }

    return {"qdrant": qd, "graph": graph, "finetune": ft,
            "usage": db.engine_usage(), "timings": timings}


_INGEST_STATS = ROOT / "ingest_stats.json"


def _ingest_stats() -> dict:
    """Время обработки файлов из последней индексации (ingest_stats.json)."""
    if _INGEST_STATS.exists():
        try:
            return _json.loads(_INGEST_STATS.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _ingest_summary() -> dict:
    """Краткая сводка по времени обработки для разделов «Система» и «Аналитика+»."""
    ist = _ingest_stats()
    lr = ist.get("last_run") or {}
    return {
        "last_duration_sec": lr.get("duration_sec"),
        "files_processed": lr.get("files_processed"),
        "chunks": lr.get("chunks"),
        "avg_ms": lr.get("avg_ms"),
        "total_ms": ist.get("total_ms"),
        "timed_files": ist.get("total_files_timed"),
        "updated": ist.get("updated"),
    }


def files_catalog(limit: int = 100, offset: int = 0, query: str = "",
                  sort: str = "name", order: str = "asc",
                  only_errors: bool = False, method: str = "") -> dict:
    """Расширенный каталог документов: файлы папки знаний с размером, датой,
    числом чанков, статусом индексации, временем обработки и способом извлечения
    текста (transcribed/ocr/tool/text); сводка по типам.
    Поддерживает пагинацию (limit/offset), сортировку
    (sort=name|date|size|chunks|proc, order=asc|desc), фильтр файлов с ошибками и
    фильтр по способу (method=transcribed|recognized|ocr|tool|text)."""
    pg = _catalog_pg_active()
    docs = Path(settings.get("DOCS_DIR")).expanduser()
    if not pg and not docs.exists():
        return {"ok": False, "msg": f"папка документов не найдена: {docs}"}

    # карта «файл -> проблема» из последней завершённой проверки каталога
    cres = _check_job.get("results") or {}
    err_map = {p.get("path"): p.get("issue") for p in (cres.get("problems") or [])}
    checked = bool(cres) and _check_job.get("ok") is True
    err_truncated = (cres.get("problems_total") or 0) > len(cres.get("problems") or [])

    # время обработки по файлам из последней индексации
    istats = _ingest_stats()
    proc_map = {k: (v.get("ms") if isinstance(v, dict) else None)
                for k, v in (istats.get("files") or {}).items()}

    # число чанков по каждому источнику — одним фасет-запросом к Qdrant.
    # Тяжёлый запрос кэшируется (Redis, пространство index — сбрасывается переиндексацией).
    def _facet():
        out = {}
        try:
            base, coll = settings.get("QDRANT_URL"), settings.get("QDRANT_COLLECTION")
            r = httpx.post(f"{base}/collections/{coll}/facet",
                           json={"key": "source", "limit": 100000, "exact": True},
                           timeout=30)
            if r.status_code == 200:
                for h in (r.json().get("result", {}) or {}).get("hits", []):
                    out[h.get("value")] = h.get("count", 0)
        except Exception:
            pass
        return out

    try:
        import cache
        counts = cache.get_or_set("facet:" + str(settings.get("QDRANT_COLLECTION")),
                                  60, _facet, ns="index")
    except Exception:
        counts = _facet()

    files, by_ext = [], {}
    total_size = indexed = 0
    if pg:
        # источник — таблица doc_catalog в PostgreSQL
        for r in db.catalog_rows():
            rel = r.get("rel_path") or ""
            ext = (r.get("ext") or "").lstrip(".")
            sz = int(r.get("size") or 0)
            mt = int(r.get("mtime") or 0)
            ch = counts.get(rel, 0)
            if ch:
                indexed += 1
            total_size += sz
            by_ext[ext] = by_ext.get(ext, 0) + 1
            meth = r.get("method") or _file_method("." + ext if ext else "")
            files.append({"path": rel, "ext": ext, "size": sz, "mtime": mt,
                          "chunks": ch, "indexed": bool(ch),
                          "error": err_map.get(rel), "proc_ms": proc_map.get(rel),
                          "method": meth})
    else:
        for p in sorted(fsutil.iter_doc_files(docs, _SUPPORTED)):
            rel = str(p.relative_to(docs))
            ext = p.suffix.lower().lstrip(".")
            try:
                sz = p.stat().st_size
                mt = int(p.stat().st_mtime)
            except Exception:
                sz, mt = 0, 0
            ch = counts.get(rel, 0)
            if ch:
                indexed += 1
            total_size += sz
            by_ext[ext] = by_ext.get(ext, 0) + 1
            meth = _file_method(p.suffix.lower())
            files.append({"path": rel, "ext": ext, "size": sz, "mtime": mt,
                          "chunks": ch, "indexed": bool(ch),
                          "error": err_map.get(rel), "proc_ms": proc_map.get(rel),
                          "method": meth})

    total = len(files)
    error_count = sum(1 for f in files if f["error"])
    transcribed_count = sum(1 for f in files if f["method"] == "transcribed")
    recognized_count = sum(1 for f in files if f["method"] in ("ocr", "tool"))
    # суммарное время обработки по всем известным файлам (мс) и сводка последнего прогона
    total_proc_ms = sum(v for v in proc_map.values() if isinstance(v, (int, float)))
    last_run = istats.get("last_run") or {}
    if query:
        ql = query.lower()
        files = [f for f in files if ql in f["path"].lower()]
    if only_errors:
        files = [f for f in files if f["error"]]
    if method:
        if method == "recognized":
            files = [f for f in files if f["method"] in ("ocr", "tool")]
        elif method in ("transcribed", "ocr", "tool", "text"):
            files = [f for f in files if f["method"] == method]
    matched = len(files)

    # сортировка
    keymap = {"name": lambda f: f["path"].lower(),
              "date": lambda f: f["mtime"],
              "size": lambda f: f["size"],
              "chunks": lambda f: f["chunks"],
              "proc": lambda f: (f["proc_ms"] if f["proc_ms"] is not None else -1),
              "error": lambda f: (0 if f["error"] else 1, f["path"].lower())}
    keyfn = keymap.get(sort, keymap["name"])
    files.sort(key=keyfn, reverse=(order == "desc"))

    offset = max(0, offset)
    limit = max(1, min(limit, 1000))
    page = files[offset:offset + limit]
    return {"ok": True, "total": total, "matched": matched, "indexed": indexed,
            "not_indexed": total - indexed, "total_size": total_size,
            "error_count": error_count, "checked": checked,
            "err_truncated": err_truncated,
            "transcribed_count": transcribed_count,
            "recognized_count": recognized_count,
            "total_proc_ms": total_proc_ms, "timed_files": len(proc_map),
            "last_run": last_run,
            "by_ext": by_ext, "files": page,
            "dir": "PostgreSQL · doc_catalog" if pg else str(docs),
            "source": "postgresql" if pg else "filesystem",
            "offset": offset, "limit": limit, "sort": sort, "order": order,
            "only_errors": only_errors, "method": method}


def file_text(source: str, max_chars: int = 20000) -> dict:
    """Извлечённый текст файла (транскрипция/распознанное/прочее) из Qdrant —
    для просмотра «в раскрытии» строки каталога. Собирает чанки по source."""
    source = (source or "").strip()
    if not source:
        return {"ok": False, "msg": "не указан файл"}
    # источник — PostgreSQL: отдаём сохранённый текст из doc_catalog
    if _catalog_pg_active():
        r = db.catalog_text(source, max_chars)
        if r is None:
            return {"ok": True, "source": source, "text": "", "chunks": 0,
                    "method": _file_method(Path(source).suffix),
                    "note": "файл отсутствует в каталоге PostgreSQL"}
        return {"ok": True, "source": source, "text": r["text"],
                "chunks": None, "method": r.get("method") or _file_method(Path(source).suffix),
                "n_chars": r.get("n_chars"), "truncated": r.get("truncated"),
                "from": "postgresql"}
    base, coll = settings.get("QDRANT_URL"), settings.get("QDRANT_COLLECTION")
    points, next_off = [], None
    try:
        for _ in range(40):  # до ~10k чанков на файл
            body = {"filter": {"must": [{"key": "source",
                                         "match": {"value": source}}]},
                    "limit": 256, "with_payload": True, "with_vector": False}
            if next_off is not None:
                body["offset"] = next_off
            r = httpx.post(f"{base}/collections/{coll}/points/scroll",
                           json=body, timeout=20)
            if r.status_code != 200:
                return {"ok": False, "msg": f"Qdrant HTTP {r.status_code}"}
            res = r.json().get("result", {}) or {}
            points.extend(res.get("points", []))
            next_off = res.get("next_page_offset")
            if next_off is None or len(points) >= 4000:
                break
    except Exception as e:
        return {"ok": False, "msg": str(e)}

    if not points:
        return {"ok": True, "source": source, "text": "",
                "chunks": 0, "method": _file_method(Path(source).suffix),
                "note": "файл не проиндексирован или текст не извлечён"}

    def _pg(p):
        v = (p.get("payload") or {}).get("page")
        return v if isinstance(v, int) else 10 ** 9
    points.sort(key=_pg)

    parts, total, truncated = [], 0, False
    for p in points:
        t = ((p.get("payload") or {}).get("text") or "").strip()
        if not t:
            continue
        if total + len(t) > max_chars:
            parts.append(t[:max(0, max_chars - total)])
            truncated = True
            break
        parts.append(t)
        total += len(t) + 2
    return {"ok": True, "source": source, "text": "\n\n".join(parts),
            "chunks": len(points), "method": _file_method(Path(source).suffix),
            "truncated": truncated or len(points) >= 4000}


def save_uploaded_folder(items: list) -> dict:
    """Сохранить загруженную целиком папку в DOCS_DIR с сохранением структуры.
    items: список (relpath, bytes). Поддерживается до десятков тысяч файлов
    (вызывается батчами из веб-интерфейса). Небезопасные пути отбрасываются."""
    docs = Path(settings.get("DOCS_DIR")).expanduser()
    docs.mkdir(parents=True, exist_ok=True)
    docs_res = docs.resolve()
    saved = skipped = 0
    bad = []
    saved_paths = []
    for rel, data in items:
        clean = [seg for seg in str(rel).replace("\\", "/").split("/")
                 if seg not in ("", ".", "..")]
        if not clean:
            skipped += 1
            continue
        ext = ("." + clean[-1].rsplit(".", 1)[-1].lower()) if "." in clean[-1] else ""
        if ext not in _SUPPORTED:
            skipped += 1
            continue
        target = (docs / Path(*clean)).resolve()
        # защита от выхода за пределы DOCS_DIR
        if docs_res not in target.parents and target != docs_res:
            bad.append("/".join(clean))
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            saved += 1
            saved_paths.append(target)
        except Exception as e:
            bad.append(f"{'/'.join(clean)} ({e})")
    # если активен каталог PostgreSQL — кладём загруженные файлы и в него
    catalog_added = catalog_add_paths(saved_paths)
    return {"ok": True, "saved": saved, "skipped": skipped,
            "errors": bad[:50], "dir": str(docs), "catalog_added": catalog_added}


def backup_create(scope: str) -> dict:
    """Создать резервную копию (фоном): settings|service|full."""
    if scope not in backup.SCOPES:
        return {"ok": False, "msg": "неизвестная область копирования"}
    if _backup_job["running"]:
        return {"ok": False, "msg": "копирование уже идёт"}

    def run():
        _backup_job.update(running=True, started=time.time(), finished=None, ok=None,
                           log="запуск…", label=backup.SCOPE_LABEL.get(scope, scope),
                           result={})
        try:
            r = backup.create(scope, progress=lambda d, t, n: _backup_job.update(
                log=f"упаковка {d}/{t}: {n}"))
            _backup_job["result"] = r
            _backup_job["ok"] = bool(r.get("ok") and r.get("integrity_ok"))
            if r.get("ok"):
                _backup_job["log"] = (f"готово: {r.get('name')} · файлов {r.get('files')} · "
                                      f"целостность " + ("OK" if r.get("integrity_ok") else "ОШИБКА"))
            else:
                _backup_job["log"] = r.get("msg", "ошибка")
        except Exception as e:
            _backup_job["ok"] = False
            _backup_job["log"] = str(e)
        _backup_job["running"] = False
        _backup_job["finished"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True, "msg": f"создание копии «{backup.SCOPE_LABEL.get(scope, scope)}» запущено"}


def backup_list() -> dict:
    return {"ok": True, "backups": backup.list_backups()}


def backup_delete(name: str) -> dict:
    return backup.delete(name)


def backup_verify_file(path: str) -> dict:
    """Проверить целостность загруженного архива (синхронно, без восстановления)."""
    return backup.verify(path)


def backup_restore_file(path: str) -> dict:
    """Восстановить из загруженного архива (фоном). path — временный файл."""
    if _restore_job["running"]:
        try:
            Path(path).unlink(missing_ok=True)  # не оставляем загруженный временный файл
        except Exception:
            pass
        return {"ok": False, "msg": "восстановление уже идёт"}

    def run():
        _restore_job.update(running=True, started=time.time(), finished=None,
                            ok=None, log="проверка архива…", result={})
        try:
            r = backup.restore(path, progress=lambda d, t, n: _restore_job.update(
                log=f"восстановление {d}/{t}: {n}"))
            _restore_job["result"] = r
            _restore_job["ok"] = bool(r.get("ok"))
            _restore_job["log"] = (f"восстановлено файлов: {r.get('restored')}"
                                   if r.get("ok") else r.get("msg", "ошибка"))
        except Exception as e:
            _restore_job["ok"] = False
            _restore_job["log"] = str(e)
        finally:
            try:
                Path(path).unlink(missing_ok=True)  # убираем временный загруженный файл
            except Exception:
                pass
        _restore_job["running"] = False
        _restore_job["finished"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True, "msg": "восстановление запущено"}


def backup_download_path(name: str):
    return backup.path_for(name)


def browse(path: str | None = None) -> dict:
    """Обзор папок на сервере для выбора DOCS_DIR.
    Возвращает текущий путь, родителя, список подпапок и число документов в папке."""
    try:
        base = Path(path).expanduser() if path else Path(settings.get("DOCS_DIR")).expanduser()
        if not base.exists() or not base.is_dir():
            base = Path.home()
        base = base.resolve()
    except Exception:
        base = Path.home()

    dirs = []
    n_docs = 0
    try:
        for p in sorted(base.iterdir(), key=lambda x: x.name.lower()):
            if p.name.startswith("."):
                continue
            if p.is_dir():
                dirs.append(p.name)
            elif p.suffix.lower() in _SUPPORTED:
                n_docs += 1
    except PermissionError:
        pass

    return {
        "path": str(base),
        "parent": str(base.parent),
        "dirs": dirs,
        "docs_here": n_docs,        # документов непосредственно в этой папке
    }


# ===================== База данных и кэш (копирование/миграция/Redis) =====================

_DB_JOB: dict = {"running": False, "ok": None, "log": "", "label": "",
                 "started": None, "finished": None}


def _dbjobview() -> dict:
    j = dict(_DB_JOB)
    if j["started"] and j["running"]:
        j["elapsed"] = round(time.time() - j["started"], 1)
    return j


def db_overview() -> dict:
    """Состояние БД-бэкендов + кэша Redis + статус последней операции."""
    import cache
    return {"db": db.db_status(), "cache": cache.status(), "job": _dbjobview()}


def db_test(backend: str) -> dict:
    return db.test_connection(backend)


def db_copy(target: str, migrate: bool = False) -> dict:
    """Запустить копирование/миграцию данных в target (фоном, может быть долго)."""
    if _DB_JOB["running"]:
        return {"ok": False, "msg": "операция с БД уже идёт"}
    if target not in ("sqlite", "mysql", "postgresql"):
        return {"ok": False, "msg": "неизвестная СУБД"}
    label = ("Миграция" if migrate else "Копирование") + f" → {target}"

    def run():
        _DB_JOB.update(running=True, ok=None, started=time.time(), finished=None,
                       log="", label=label)
        try:
            res = db.migrate(target) if migrate else db.copy_all(target)
            _DB_JOB["log"] = res.get("log", "") or _json.dumps(res, ensure_ascii=False)
            _DB_JOB["ok"] = bool(res.get("ok"))
        except Exception as e:
            _DB_JOB["ok"] = False
            _DB_JOB["log"] = f"ОШИБКА: {e}"
        _DB_JOB["running"] = False
        _DB_JOB["finished"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True, "msg": f"{label}: запущено"}


def cache_clear() -> dict:
    import cache
    return {"ok": True, "cleared": cache.clear()}


# ----- Каталог документов в PostgreSQL -----

_CAT_JOB: dict = {"running": False, "ok": None, "processed": 0, "total": 0,
                  "errors": 0, "stored": 0, "skipped": 0, "log": "",
                  "started": None, "finished": None}
_CAT_TEXT_CAP = 500_000              # макс. символов текста на файл (для предпросмотра)
_CAT_FILE_MAX = 100 * 1024 ** 3      # до 100 ГБ: файлы крупнее — только метаданные
_CAT_LO_MIN = 64 * 1024 * 1024       # PostgreSQL: файлы от этого размера — Large Object
_CAT_FLUSH_FILES = 100               # размер пакета записи (по числу небольших файлов)
_CAT_FLUSH_BYTES = 16 * 1024 * 1024  # либо по суммарному объёму содержимого в пакете


def _sha256_file(p) -> str:
    """SHA-256 файла потоково (без загрузки целиком в память)."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _catalog_prepare(rel, p, sz, mt, method, txt, ex):
    """Подготовить запись каталога для одного файла с учётом размера и бэкенда.

    Возвращает (status, row|None):
      - 'meta'    : файл слишком большой (> _CAT_FILE_MAX) — только метаданные (row для пакета)
      - 'stored'  : крупный файл записан потоково как Large Object (PostgreSQL)
      - 'batched' : небольшой файл — row для пакетной bytea-вставки
      - 'skipped' : содержимое не изменилось (по SHA-256)
      - 'error'   : запись Large Object не удалась
    """
    fname = p.name
    ext = p.suffix.lower().lstrip(".")
    pg = (db._dialect() == "postgresql")
    if sz > _CAT_FILE_MAX:
        return ("meta", (rel, fname, ext, sz, mt, len(txt or ""), method, "", None,
                         txt or ""))
    if pg and sz >= _CAT_LO_MIN:
        try:
            sha = _sha256_file(p)
        except Exception:
            sha = ""
        if ex and ex.get("has_content") and sha and ex.get("sha256") == sha:
            return ("skipped", None)
        ok = db.catalog_store_large_pg(rel, fname, ext, sz, mt, method, str(p), sha,
                                       txt or "")
        return ("stored" if ok else "error", None)
    try:
        content = p.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
    except Exception:
        content, sha = None, ""
    if ex and ex.get("has_content") and sha and ex.get("sha256") == sha:
        return ("skipped", None)
    return ("batched", (rel, fname, ext, sz, mt, len(txt or ""), method, sha, content,
                        txt or ""))


def _catalog_pg_active() -> bool:
    """Каталог сейчас читается из PostgreSQL (настройка включена И активна PG)."""
    return (settings.get("CATALOG_SOURCE") == "postgresql"
            and db._dialect() == "postgresql")


def catalog_status() -> dict:
    active_pg = db._dialect() == "postgresql"
    src = settings.get("CATALOG_SOURCE") or "filesystem"
    meta = db.catalog_meta() if active_pg else {"count": 0, "total_size": 0,
                                                "files_stored": 0, "updated": None,
                                                "by_ext": {}}
    job = dict(_CAT_JOB)
    if job["started"] and job["running"]:
        job["elapsed"] = round(time.time() - job["started"], 1)
    return {"pg_active": active_pg,
            "source": src if active_pg else "filesystem",
            "pg": meta, "job": job,
            "can_use_pg": active_pg and meta.get("count", 0) > 0}


def _qdrant_text_by_source(cap_per_file: int) -> dict:
    """Один проход scroll по коллекции Qdrant: собирает уже извлечённый текст по
    каждому файлу (source). Это быстро — не нужно заново парсить/OCR/транскрибировать.
    Возвращает {source: text}. Текст на файл ограничен cap_per_file символов."""
    base = settings.get("QDRANT_URL")
    coll = settings.get("QDRANT_COLLECTION")
    acc: dict = {}        # source -> [running_len, [parts]]
    next_off = None
    for _ in range(200000):  # страховка от бесконечного цикла
        body = {"limit": 512, "with_payload": ["source", "text"],
                "with_vector": False}
        if next_off is not None:
            body["offset"] = next_off
        try:
            r = httpx.post(f"{base}/collections/{coll}/points/scroll",
                           json=body, timeout=60)
        except Exception:
            break
        if r.status_code != 200:
            break
        res = r.json().get("result", {}) or {}
        for p in res.get("points", []):
            pl = p.get("payload") or {}
            src = pl.get("source")
            tx = (pl.get("text") or "").strip()
            if not src or not tx:
                continue
            cur = acc.setdefault(src, [0, []])
            if cur[0] >= cap_per_file:
                continue
            cur[1].append(tx)
            cur[0] += len(tx) + 2
        next_off = res.get("next_page_offset")
        if next_off is None:
            break
    return {s: "\n\n".join(v[1])[:cap_per_file] for s, v in acc.items()}


def catalog_load() -> dict:
    """Загрузить каталог документов в таблицу doc_catalog активной PostgreSQL — быстро.

    Текст берётся из уже построенного индекса Qdrant (без повторного парсинга/OCR/
    транскрибации), запись идёт пакетами в одном соединении. Фоновая задача."""
    if db._dialect() != "postgresql":
        return {"ok": False, "msg": "Активная БД — не PostgreSQL. Сначала мигрируйте "
                                    "на PostgreSQL в блоке «База данных и кэш»."}
    t = db.test_connection("postgresql")
    if not t.get("ok"):
        return {"ok": False, "msg": "PostgreSQL недоступна: " + t.get("msg", "")}
    if _CAT_JOB["running"]:
        return {"ok": False, "msg": "загрузка каталога уже идёт"}
    docs = Path(settings.get("DOCS_DIR")).expanduser()
    if not docs.exists():
        return {"ok": False, "msg": f"папка документов не найдена: {docs}"}

    def run():
        _CAT_JOB.update(running=True, ok=None, processed=0, total=0, errors=0,
                        stored=0, skipped=0, log="чтение индекса и каталога…",
                        started=time.time(), finished=None)
        try:
            # текст для предпросмотра — из готового индекса (быстро, без повторного парсинга)
            text_map = _qdrant_text_by_source(_CAT_TEXT_CAP)
            # что уже лежит в БД — для пропуска неизменённых файлов (по размеру/дате/sha256)
            existing = db.catalog_existing()
            _CAT_JOB["log"] = "сканирование папки…"
            paths = sorted(fsutil.iter_doc_files(docs, _SUPPORTED))
            _CAT_JOB["total"] = len(paths)

            batch = []
            batch_bytes = 0

            def _flush():
                nonlocal batch, batch_bytes
                if not batch:
                    return
                try:
                    db.catalog_store_many(batch)
                    _CAT_JOB["stored"] += len(batch)
                except Exception as e:
                    _CAT_JOB["errors"] += len(batch)
                    print(f"[catalog] пакетная запись: {e}")
                batch = []
                batch_bytes = 0

            for i, p in enumerate(paths, 1):
                _CAT_JOB["processed"] = i
                rel = str(p.relative_to(docs))
                ext = p.suffix.lower().lstrip(".")
                try:
                    stt = p.stat()
                    sz, mt = stt.st_size, int(stt.st_mtime)
                except Exception:
                    sz, mt = 0, 0
                ex = existing.get(rel)
                # быстрый пропуск: файл уже сохранён и не менялся (размер+дата)
                if ex and ex.get("has_content") and ex.get("size") == sz \
                        and ex.get("mtime") == mt:
                    _CAT_JOB["skipped"] += 1
                    continue
                txt = text_map.get(rel, "")
                status, row = _catalog_prepare(rel, p, sz, mt,
                                               _file_method(p.suffix.lower()), txt, ex)
                if status == "skipped":
                    _CAT_JOB["skipped"] += 1
                    continue
                if status == "stored":
                    _CAT_JOB["stored"] += 1
                    continue
                if status == "error":
                    _CAT_JOB["errors"] += 1
                    continue
                # 'batched' | 'meta' — кладём в пакет bytea
                batch.append(row)
                batch_bytes += len(row[8]) if row[8] else 0
                if len(batch) >= _CAT_FLUSH_FILES or batch_bytes >= _CAT_FLUSH_BYTES:
                    _flush()
                if i % 50 == 0 or i == len(paths):
                    _CAT_JOB["log"] = (
                        f"обработано {i} из {len(paths)} · сохранено "
                        f"{_CAT_JOB['stored']}, без изменений {_CAT_JOB['skipped']}, "
                        f"ошибок {_CAT_JOB['errors']}")
            _flush()
            _CAT_JOB["ok"] = True
            _CAT_JOB["log"] = (
                f"готово: всего {len(paths)} файлов, сохранено/обновлено "
                f"{_CAT_JOB['stored']}, пропущено без изменений {_CAT_JOB['skipped']}, "
                f"ошибок {_CAT_JOB['errors']}. Теперь доступна кнопка «Перейти на "
                "работу с каталогом в PostgreSQL».")
        except Exception as e:
            _CAT_JOB["ok"] = False
            _CAT_JOB["log"] = f"ОШИБКА: {e}"
        _CAT_JOB["running"] = False
        _CAT_JOB["finished"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True, "msg": "загрузка каталога в PostgreSQL запущена"}


def catalog_clear_files() -> dict:
    """Удалить из PostgreSQL только содержимое файлов (метаданные/текст остаются)."""
    if db._dialect() != "postgresql":
        return {"ok": False, "msg": "Активная БД — не PostgreSQL"}
    if _CAT_JOB["running"]:
        return {"ok": False, "msg": "идёт загрузка каталога — дождитесь завершения"}
    n = db.catalog_clear_files()
    return {"ok": True, "cleared": n,
            "msg": f"очищено файлов: {n} (метаданные и текст сохранены)"}


def catalog_add_paths(paths) -> int:
    """Если активен каталог PostgreSQL — добавить указанные файлы (целиком, с SHA-256)
    в doc_catalog. Используется при загрузке файлов/папок и парсинге сайтов, чтобы новые
    документы попадали в PostgreSQL и учитывались при индексации из БД (без папки).
    paths — пути внутри DOCS_DIR. Возвращает число добавленных записей."""
    if not _catalog_pg_active():
        return 0
    docs = Path(settings.get("DOCS_DIR")).expanduser()
    rows: list = []
    nbytes = 0
    added = 0

    def _flush():
        nonlocal rows, nbytes, added
        if not rows:
            return
        try:
            db.catalog_store_many(rows)
            added += len(rows)
        except Exception as e:
            print(f"[catalog] добавление файлов: {e}")
        rows = []
        nbytes = 0

    for raw in paths:
        p = Path(raw)
        try:
            rel = str(p.relative_to(docs))
        except Exception:
            rel = p.name
        try:
            sz = p.stat().st_size
            mt = int(p.stat().st_mtime)
        except Exception:
            continue
        status, row = _catalog_prepare(rel, p, sz, mt, _file_method(p.suffix.lower()),
                                       "", None)
        if status == "stored":
            added += 1
            continue
        if status == "error":
            continue
        # 'batched' | 'meta'
        rows.append(row)
        nbytes += len(row[8]) if row[8] else 0
        if len(rows) >= _CAT_FLUSH_FILES or nbytes >= _CAT_FLUSH_BYTES:
            _flush()
    _flush()
    if added:
        print(f"[catalog] в PostgreSQL добавлено/обновлено файлов: {added}")
    return added


def catalog_use(source: str) -> dict:
    """Переключить источник каталога: postgresql | filesystem."""
    source = "postgresql" if source == "postgresql" else "filesystem"
    if source == "postgresql":
        if db._dialect() != "postgresql":
            return {"ok": False, "msg": "Активная БД — не PostgreSQL"}
        if db.catalog_count() <= 0:
            return {"ok": False, "msg": "Каталог в PostgreSQL пуст — сначала загрузите его"}
    settings.update({"CATALOG_SOURCE": source})
    return {"ok": True, "source": source,
            "msg": ("Каталог документов теперь читается из PostgreSQL"
                    if source == "postgresql"
                    else "Каталог документов снова читается из папки (файловая система)")}


def _update_env(path: Path, kv: dict) -> None:
    lines = path.read_text().splitlines() if path.exists() else []
    seen = set()
    out = []
    for ln in lines:
        key = ln.split("=", 1)[0] if "=" in ln else None
        if key in kv:
            out.append(f"{key}={kv[key]}")
            seen.add(key)
        else:
            out.append(ln)
    for k, v in kv.items():
        if k not in seen:
            out.append(f"{k}={v}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")

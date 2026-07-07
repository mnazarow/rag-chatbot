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
import vectorstore
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


def _read_full_log(path: str, cap: int = 5_000_000) -> str:
    """Полный текст лог-файла (с ограничением размера)."""
    try:
        t = Path(path).read_text(errors="ignore")
        return t[-cap:] if len(t) > cap else t
    except Exception:
        return ""


def _bg(job: dict, label: str, cmds: list, logfile: str, timeout: int = 24 * 3600,
        save_label: str | None = None) -> dict:
    """Запустить команды последовательно в фоне, вывод — в logfile (живой лог).
    cmds: список команд (каждая — list аргументов). save_label — если задан, по
    завершении полный лог сохраняется в БД (таблица ingest_logs)."""
    if job["running"]:
        return {"ok": False, "msg": f"{label}: задача уже идёт"}

    def run():
        job.update(running=True, started=time.time(), finished=None, ok=None,
                   log="", logfile=logfile, stopped=False, _proc=None)
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
                    # start_new_session — чтобы можно было прибить всю группу процессов
                    # (ingest.py + его воркеры) при остановке
                    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fp,
                                            stderr=subprocess.STDOUT,
                                            start_new_session=True)
                    job["_proc"] = proc
                    rc = proc.wait(timeout=timeout)
                    job["_proc"] = None
                    if rc != 0:
                        ok = False
                        break
            job["log"] = _tail(logfile)
            if job.get("stopped"):
                ok = False
                job["log"] = (job["log"] + "\n[остановлено пользователем]").strip()
            if "summary" in job:
                job["summary"] = _extract_summary(job["log"])
            job["ok"] = ok
        except Exception as e:
            job["ok"] = False
            job["log"] = (_tail(logfile) + "\n" + str(e)).strip()
        job["running"] = False
        job["finished"] = time.time()
        # сохраняем полный лог в БД (для просмотра/удаления в админке)
        if save_label:
            try:
                db.ingest_log_save(save_label, job.get("summary") or "",
                                   _read_full_log(logfile))
            except Exception as e:
                print(f"[bg] не удалось сохранить лог в БД: {e}")

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


# Фоновые задачи для раздела «Текущие запросы» (только идущие + недавно завершённые)
_JOB_META = [
    ("index", "Индексация документов", "🗂"),
    ("finetune", "Дообучение модели", "🎓"),
    ("graph", "Построение графа знаний", "🕸"),
    ("web", "Парсинг сайтов", "🌐"),
    ("bench", "Бенчмарк", "⚡"),
    ("test", "Самотестирование", "🧪"),
    ("check", "Проверка данных", "🔍"),
    ("backup", "Резервная копия", "💾"),
    ("restore", "Восстановление", "♻️"),
    ("pull", "Загрузка модели", "⬇️"),
    ("dep", "Установка зависимостей", "📦"),
]


def active_jobs(recent_sec: float = 10.0) -> list[dict]:
    """Идущие (и только что завершённые) фоновые задачи — для дашборда."""
    jobs = {
        "index": _job, "finetune": _ft_job, "graph": _graph_job, "web": _web_job,
        "bench": _bench_job, "test": _test_job, "check": _check_job,
        "backup": _backup_job, "restore": _restore_job, "pull": _pull_job,
        "dep": _dep_job,
    }
    now = time.time()
    out = []
    for key, label, icon in _JOB_META:
        jb = jobs.get(key) or {}
        running = bool(jb.get("running"))
        fin = jb.get("finished")
        if not running and not (fin and (now - fin) < recent_sec):
            continue
        started = jb.get("started") or now
        end = now if running else (fin or now)
        summary = (jb.get("summary") or "").strip()
        out.append({
            "kind": "job", "job": key, "label": label, "icon": icon,
            "running": running, "ok": jb.get("ok"),
            "stage": summary[:140] or ("выполняется…" if running else "завершено"),
            "elapsed_ms": int((end - started) * 1000),
        })
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
_WEB_STATS = ROOT / "web_stats.json"    # результаты парсинга по каждому сайту (ошибки, лимиты)
_web_dl_lock = threading.Lock()         # защита выбора имени файла при параллельном скачивании


# Список сайтов и статистику парсинга храним в БД (таблица kv_store), чтобы они
# переживали пересоздание контейнера в Docker: файлы в /app не персистентны, а
# rag_logs.db смонтирован. Старые файлы web_sources.txt/web_stats.json один раз
# мигрируются в БД (для нативных установок, где они уже есть).

def _web_sources_load() -> list:
    """URL сохранённых сайтов из БД (с одноразовой миграцией из web_sources.txt)."""
    raw = db.kv_get("web_sources")
    if raw is None and _WEB_SOURCES.exists():
        try:
            raw = _WEB_SOURCES.read_text(encoding="utf-8")
            db.kv_set("web_sources", raw)
        except Exception:
            raw = ""
    return [u.strip() for u in (raw or "").splitlines() if u.strip()]


def _web_sources_save(urls: list) -> None:
    try:
        db.kv_set("web_sources", "\n".join(urls))
    except Exception as e:
        print(f"[web] не удалось сохранить список сайтов: {e}")


def _web_stats_load() -> dict:
    """Результаты парсинга по сайтам из БД (с миграцией из web_stats.json)."""
    raw = db.kv_get("web_stats")
    if raw is None and _WEB_STATS.exists():
        try:
            raw = _WEB_STATS.read_text(encoding="utf-8")
            db.kv_set("web_stats", raw)
        except Exception:
            raw = ""
    try:
        return _json.loads(raw) if raw else {}
    except Exception:
        return {}


def _web_stats_save(stats: dict) -> None:
    try:
        db.kv_set("web_stats", _json.dumps(stats, ensure_ascii=False))
    except Exception as e:
        print(f"[web] не удалось сохранить статистику парсинга: {e}")


def _web_excludes_load() -> dict:
    """Исключения по ключевым словам для каждого сайта: {url: [kw, ...]} из БД."""
    raw = db.kv_get("web_excludes")
    try:
        d = _json.loads(raw) if raw else {}
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _web_excludes_save(m: dict) -> None:
    try:
        db.kv_set("web_excludes", _json.dumps(m, ensure_ascii=False))
    except Exception as e:
        print(f"[web] не удалось сохранить исключения: {e}")


def _web_parse_keywords(val) -> list:
    """Строку/список ключевых слов → нормализованный список (нижний регистр, без дублей)."""
    import re
    if isinstance(val, str):
        parts = re.split(r"[\s,;\n]+", val)
    else:
        parts = list(val or [])
    out, seen = [], set()
    for p in parts:
        k = str(p).strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out[:50]


def _web_excluded(url: str, excludes) -> bool:
    """URL исключён, если содержит любое из ключевых слов (без учёта регистра)."""
    if not excludes:
        return False
    u = (url or "").lower()
    return any(kw in u for kw in excludes if kw)


def web_set_excludes(url: str, keywords) -> dict:
    """Задать/обновить список исключений по ключевым словам для сайта. Применяется при
    следующем парсинге этого сайта (ручном или по расписанию)."""
    url = (url or "").strip()
    if not url:
        return {"ok": False, "msg": "не указан URL сайта"}
    kws = _web_parse_keywords(keywords)
    m = _web_excludes_load()
    if kws:
        m[url] = kws
    else:
        m.pop(url, None)
    _web_excludes_save(m)
    return {"ok": True, "url": url, "exclude": kws,
            "msg": (f"исключения сохранены ({len(kws)} слов) — применятся при следующем "
                    "парсинге этого сайта") if kws else "исключения очищены"}


def _web_slug(url: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9._-]", "_", url)[:120] or "page"


def _web_list() -> list:
    """Список сохранённых сайтов: URL + файл + признак наличия в папке."""
    urls = _web_sources_load()
    webdir = Path(settings.get("DOCS_DIR")).expanduser() / "web"
    stats = _web_stats_load()
    excl = _web_excludes_load()
    out = []
    for u in urls:
        rel = f"web/{_web_slug(u)}.html"
        st = stats.get(u, {})
        out.append({"url": u, "source": rel,
                    "indexed": (webdir / f"{_web_slug(u)}.html").exists(),
                    "ok": st.get("ok"), "pages": st.get("pages"), "files": st.get("files"),
                    "errors": st.get("errors", []), "limits": st.get("limits", {}),
                    "ts": st.get("ts"),
                    "exclude": excl.get(u, []),
                    "progress": st.get("progress"),
                    "n_items": len(st.get("items") or [])})
    return out


def web_saved_urls() -> list:
    """Список сохранённых для парсинга сайтов (URL)."""
    return _web_sources_load()


def web_structure() -> dict:
    """Структура спарсенных сайтов в реальном времени: по каждому сайту — прогресс
    и дерево элементов (объединённый документ страниц + скачанные файлы) с признаком
    добавления в БД, числом чанков, способом распознавания, наличием LLM-описания,
    типом, размером, датой изменения и временем обработки. Тяжёлые данные (чанки,
    описания, время обработки) — из тех же источников, что каталог документов."""
    docs = Path(settings.get("DOCS_DIR")).expanduser()
    stats = _web_stats_load()
    urls = _web_sources_load()

    # число чанков и описания по источникам — одним фасет-запросом (кэш index, 15 c)
    def _facets():
        chunks, described = {}, set()
        try:
            for h in vectorstore.facet("source", 100000):
                chunks[h.get("value")] = h.get("count", 0)
            for h in vectorstore.facet("source", 100000, flt={"vision_desc": True}):
                if h.get("value"):
                    described.add(h.get("value"))
        except Exception:
            pass
        return {"chunks": chunks, "described": list(described)}
    try:
        import cache
        fac = cache.get_or_set("webstruct_facet:" + str(vectorstore.backend()) + ":"
                               + str(settings.get("QDRANT_COLLECTION")),
                               15, _facets, ns="index")
    except Exception:
        fac = _facets()
    chunk_map = fac.get("chunks", {})
    desc_set = set(fac.get("described", []))

    # время обработки по файлам из последней индексации
    try:
        proc_map = {k: (v.get("ms") if isinstance(v, dict) else None)
                    for k, v in (_ingest_stats().get("files") or {}).items()}
    except Exception:
        proc_map = {}

    sites = []
    for u in urls:
        st = stats.get(u, {})
        items = []
        for it in (st.get("items") or []):
            src = it.get("source")
            ext = Path(src).suffix if src else ""
            n_ch = chunk_map.get(src, 0) if src else 0
            fpath = (docs / src) if src else None
            mtime = None
            try:
                if fpath and fpath.exists():
                    mtime = fpath.stat().st_mtime
            except Exception:
                pass
            items.append({
                "kind": it.get("kind"), "name": it.get("name"), "url": it.get("url"),
                "source": src, "ok": it.get("ok", True),
                "size": it.get("size", 0), "type": ext.lstrip(".").upper() or "HTML",
                "method": _file_method(ext) if ext else "text",
                "chunks": n_ch, "in_db": n_ch > 0,
                "described": (src in desc_set), "mtime": mtime,
                "proc_ms": proc_map.get(src),
                "pages": it.get("pages"),
            })
        sites.append({"url": u, "ok": st.get("ok"), "ts": st.get("ts"),
                      "progress": st.get("progress") or {"phase": "—", "pct": 0},
                      "errors": st.get("errors", []), "limits": st.get("limits", {}),
                      "items": items})
    return {"sites": sites, "running": bool(_web_job.get("running"))}


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
    _web_sources_save(sites)
    # убрать сохранённую статистику парсинга по этому сайту
    _st = _web_stats_load()
    if url in _st:
        _st.pop(url, None)
        _web_stats_save(_st)
    # убрать исключения по ключевым словам для этого сайта
    _ex = _web_excludes_load()
    if url in _ex:
        _ex.pop(url, None)
        _web_excludes_save(_ex)
    # 2) удалить файл
    slug = _web_slug(url)
    f = Path(settings.get("DOCS_DIR")).expanduser() / "web" / f"{slug}.html"
    if f.exists():
        try:
            f.unlink()
        except Exception:
            pass
    # 3) удалить чанки из векторной базы (Qdrant/Milvus) по source
    src = f"web/{slug}.html"
    try:
        vectorstore.delete({"source": src})
    except Exception as e:
        return {"ok": True, "msg": f"удалён из списка и папки; из векторной базы не удалён: {e}"}
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


def _fmt_bytes(n: int) -> str:
    n = float(n or 0)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if n < 1024 or unit == "ГБ":
            return f"{n:.0f} {unit}" if unit == "Б" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} ГБ"


def _web_download(url: str, dest_dir, log):
    """Скачать файл по ссылке (любого типа) в dest_dir потоково, с процентом загрузки.
    Возвращает путь или None."""
    import re
    from urllib.parse import urlparse, unquote
    tmp = None
    try:
        name = unquote(os.path.basename(urlparse(url).path)) or _web_slug(url)
        name = re.sub(r"[^\w.\-]+", "_", name)[:150] or "file"
        dest_dir.mkdir(parents=True, exist_ok=True)
        # выбор уникального имени + резервирование .part — под локом (безопасно при
        # параллельном скачивании: два потока не займут одно имя)
        with _web_dl_lock:
            out = dest_dir / name
            i = 1
            while out.exists() or out.with_name(out.name + ".part").exists():
                out = dest_dir / f"{i}_{name}"
                i += 1
            tmp = out.with_name(out.name + ".part")
            tmp.touch()
        with httpx.stream("GET", url, timeout=120, follow_redirects=True,
                          headers={"User-Agent": "Mozilla/5.0 (RAGBot)"}) as r:
            if r.status_code != 200:
                log(f"    ERR файл {url}: HTTP {r.status_code}")
                return None
            total = int(r.headers.get("content-length") or 0)
            got, last = 0, -20
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(256 * 1024):
                    f.write(chunk)
                    got += len(chunk)
                    if total:
                        pct = int(got * 100 / total)
                        if pct >= last + 20:          # отметки 0/20/40/60/80/100%
                            last = pct
                            log(f"        {name}: {pct}% "
                                f"({_fmt_bytes(got)} из {_fmt_bytes(total)})")
        tmp.rename(out)
        log(f"    ФАЙЛ сохранён: {out.name} ({_fmt_bytes(got)})")
        return out
    except Exception as e:
        try:
            if tmp is not None:
                tmp.unlink()
        except Exception:
            pass
        log(f"    ERR файл {url}: {e}")
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


def _web_crawl(seed: str, depth: int, max_pages: int, same_domain: bool, renderer, log,
               workers: int = 1, excludes=None):
    """Обойти сайт из стартовой страницы (BFS по уровням) до depth/max_pages. Возвращает
    (pages, files, stat): pages — список (url, title, text); files — множество URL
    файлов для скачивания; stat — {errors, pages_limit_hit, depth_limit_hit, excluded}.
    excludes — список ключевых слов: URL, содержащие любое из них, пропускаются.
    При workers>1 и без headless-браузера страницы уровня грузятся параллельно."""
    from urllib.parse import urlparse
    from concurrent.futures import ThreadPoolExecutor
    seed_netloc = urlparse(seed).netloc
    seen, level, pages, files = set(), [(seed, 0)], [], set()
    errors = []
    depth_limit_hit = False
    excluded_n = 0
    # параллелим только обычную загрузку; с рендерером (Playwright) — последовательно
    par = max(1, int(workers or 1)) if renderer is None else 1

    while level and len(pages) < max_pages:
        # отобрать новые URL уровня; прямые ссылки на файлы отложить в files
        batch = []
        for url, d in level:
            if url in seen:
                continue
            seen.add(url)
            if _web_excluded(url, excludes):
                excluded_n += 1
                continue
            if _web_is_file(url):
                files.add(url)
            else:
                batch.append((url, d))
        if not batch:
            break

        # загрузка страниц уровня (параллельно или последовательно)
        fetched = []   # (url, d, html)
        if par > 1 and len(batch) > 1:
            with ThreadPoolExecutor(max_workers=min(par, len(batch))) as ex:
                futs = {ex.submit(_web_fetch, u, None, log): (u, dd) for (u, dd) in batch}
                for fu in futs:
                    u, dd = futs[fu]
                    try:
                        fetched.append((u, dd, fu.result()))
                    except Exception as e:
                        log(f"ERR {u}: {e}")
                        fetched.append((u, dd, None))
        else:
            for (u, dd) in batch:
                fetched.append((u, dd, _web_fetch(u, renderer, log)))

        next_level = []
        for url, d, html in fetched:
            if len(pages) >= max_pages:
                break
            if html is None:
                errors.append(f"страница не загружена: {url}")
                continue
            text = _web_extract(html)
            if text:
                pages.append((url, _web_title(html, url), text))
                log(f"  стр. {len(pages)}/{max_pages}: {url}  ({len(text)} симв.)")
            else:
                log(f"  стр. (без текста): {url}")
                errors.append(f"страница без извлечённого текста (возможно, JS-сайт): {url}")
            for link in _web_links(html, url, seed_netloc, same_domain):
                if link in seen:
                    continue
                if _web_excluded(link, excludes):
                    excluded_n += 1
                    continue
                if _web_is_file(link):
                    files.add(link)
                elif d < depth:
                    next_level.append((link, d + 1))
                else:
                    depth_limit_hit = True   # были более глубокие ссылки, но глубина не позволяет
        level = next_level

    pages_limit_hit = bool(level) and len(pages) >= max_pages
    return pages, files, {"errors": errors, "pages_limit_hit": pages_limit_hit,
                          "depth_limit_hit": depth_limit_hit, "excluded": excluded_n}


def ingest_web(urls: list, index: bool = True) -> dict:
    """Скачать до 50 сайтов (с обходом ссылок), извлечь текст в DOCS_DIR/web и
    (при index=True) переиндексировать. При index=False — только парсинг без
    индексации (быстро обновить содержимое; проиндексировать можно позже кнопкой
    «Переиндексировать»). Глубина/лимит/домен — настройки WEB_CRAWL_DEPTH/MAX_PAGES/SAME_DOMAIN."""
    import re
    urls = [u.strip() for u in (urls or []) if u.strip().startswith(("http://", "https://"))][:50]
    if not urls:
        return {"ok": False, "msg": "укажите хотя бы один URL (http/https), максимум 50"}
    if _web_job["running"]:
        return {"ok": False, "msg": "парсинг сайтов уже идёт"}
    _web_sources_save(urls)
    excludes_map = _web_excludes_load()
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
            workers = max(1, int(settings.get("WEB_CONCURRENCY") or 1))
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
                stats_map = {}
                try:
                    for u in urls:
                        site_errors, site_limits, site_items = [], {}, []
                        pages, dl, site_ok = [], 0, False
                        stats_map[u] = {"url": u, "ts": time.time(), "items": [],
                                        "progress": {"phase": "crawl", "pct": 8}}
                        _web_stats_save(stats_map)
                        try:
                            _log(f"САЙТ: {u}")
                            site_excludes = excludes_map.get(u) or []
                            if site_excludes:
                                _log(f"  исключения по словам: {', '.join(site_excludes)}")
                            pages, file_urls, cstat = _web_crawl(u, depth, max_pages,
                                                                 same_domain, renderer, _log,
                                                                 workers, excludes=site_excludes)
                            site_errors.extend(cstat.get("errors", []))
                            if cstat.get("excluded"):
                                _log(f"  исключено URL по ключевым словам: {cstat['excluded']}")
                            if cstat.get("pages_limit_hit"):
                                site_limits["pages"] = (f"достигнут лимит страниц ({max_pages}); "
                                                        "часть страниц не обойдена — увеличьте «Макс. страниц»")
                            if cstat.get("depth_limit_hit"):
                                site_limits["depth"] = (f"достигнута глубина обхода ({depth}); "
                                                        "более глубокие ссылки пропущены — увеличьте «Глубину обхода»")
                            # скачиваем найденные файлы (любого типа), лимит на сайт
                            all_files = [f for f in file_urls
                                         if not _web_excluded(f, site_excludes)]
                            if len(all_files) > max_files:
                                site_limits["files"] = (f"найдено файлов {len(all_files)}, скачано {max_files} "
                                                        f"(лимит); {len(all_files) - max_files} пропущено — "
                                                        "увеличьте «Макс. файлов»")
                            file_list = all_files[:max_files]
                            nf = len(file_list)
                            _docroot = Path(settings.get("DOCS_DIR")).expanduser()
                            stats_map[u]["progress"] = {"phase": "download", "done": 0, "total": nf, "pct": 40}
                            _web_stats_save(stats_map)
                            if nf:
                                _log(f"  файлов к скачиванию: {nf}"
                                     + (f" (параллельно ×{workers})" if workers > 1 and nf > 1 else ""))

                            def _rec_file(furl, p):
                                okf = p is not None
                                rel, size = None, 0
                                if okf:
                                    try:
                                        rel = str(p.relative_to(_docroot))
                                        size = p.stat().st_size
                                    except Exception:
                                        rel = str(p)
                                site_items.append({"kind": "file", "url": furl,
                                                   "name": (p.name if okf else (furl.rsplit("/", 1)[-1] or furl)),
                                                   "source": rel, "size": size, "ok": okf})
                                done = len(site_items)
                                stats_map[u]["progress"] = {"phase": "download", "done": done, "total": nf,
                                                            "pct": min(88, 40 + int(done * 45 / max(1, nf)))}
                                if done % 5 == 0:
                                    _web_stats_save(stats_map)

                            if workers > 1 and nf > 1:
                                # параллельное скачивание файлов (потокобезопасно: httpx.stream + уникальные имена)
                                from concurrent.futures import ThreadPoolExecutor
                                done_n = [0]
                                def _dl_one(furl):
                                    p = _web_download(furl, filesdir, _log)
                                    done_n[0] += 1
                                    _log(f"  [файл {done_n[0]}/{nf}] {int(done_n[0]*100/nf)}%: {furl}"
                                         + ("" if p is not None else "  — не скачан"))
                                    return furl, p
                                with ThreadPoolExecutor(max_workers=min(workers, nf)) as ex:
                                    for furl, p in ex.map(_dl_one, file_list):
                                        if p is not None:
                                            web_paths.append(p)
                                            dl += 1
                                        else:
                                            site_errors.append(f"файл не скачан: {furl}")
                                        _rec_file(furl, p)
                            else:
                                for fi, furl in enumerate(file_list, 1):
                                    fpct = int(fi * 100 / nf) if nf else 100
                                    _log(f"  [файл {fi}/{nf}] {fpct}% скачиваю: {furl}")
                                    p = _web_download(furl, filesdir, _log)
                                    if p is not None:
                                        web_paths.append(p)
                                        dl += 1
                                    else:
                                        site_errors.append(f"файл не скачан: {furl}")
                                    _rec_file(furl, p)
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
                                try:
                                    prel = str(out.relative_to(_docroot))
                                    psize = out.stat().st_size
                                except Exception:
                                    prel, psize = str(out), 0
                                site_items.insert(0, {"kind": "page", "name": "Страницы сайта (объединено)",
                                                      "source": prel, "size": psize, "ok": True,
                                                      "pages": len(pages)})
                            if pages or dl:
                                ok += 1
                                site_ok = True
                                fp.write(f"ИТОГО {u}: страниц {len(pages)}, файлов {dl}\n")
                            else:
                                err += 1
                                site_errors.append("ни текста, ни файлов (пустая страница "
                                                   "или JS-сайт без headless-браузера)")
                                fp.write(f"ERR {u}: ни текста, ни файлов (пустая страница "
                                         "или JS-сайт без headless-браузера)\n")
                        except Exception as e:
                            err += 1
                            site_errors.append(str(e))
                            fp.write(f"ERR {u}: {e}\n")
                        stats_map[u] = {"url": u, "ok": site_ok, "pages": len(pages),
                                        "files": dl, "errors": site_errors,
                                        "limits": site_limits, "ts": time.time(),
                                        "items": site_items,
                                        "progress": {"phase": "parsed", "pct": 92}}
                        _web_stats_save(stats_map)
                        fp.flush()
                finally:
                    if renderer is not None:
                        renderer.close()
                    _web_stats_save(stats_map)
                # активен каталог PostgreSQL — кладём спарсенные страницы и в него,
                # чтобы индексация из БД их увидела (без папки)
                added = catalog_add_paths(web_paths)
                if added:
                    fp.write(f"В PostgreSQL добавлено страниц: {added}\n")
                def _set_all_progress(prog):
                    for _u in stats_map:
                        stats_map[_u]["progress"] = dict(prog)
                    _web_stats_save(stats_map)

                if index:
                    _set_all_progress({"phase": "index", "pct": 96})
                    fp.write(f"Скачано: {ok}, ошибок: {err}. Запускаю индексацию...\n")
                    fp.flush()
                    rc = subprocess.Popen([sys.executable, "-u", "ingest.py"], cwd=ROOT,
                                          stdout=fp, stderr=subprocess.STDOUT).wait(timeout=24 * 3600)
                    fp.write(f"SUMMARY web_ok={ok} web_err={err} index_rc={rc}\n")
                    _set_all_progress({"phase": "done", "pct": 100})
                else:
                    rc = 0
                    fp.write(f"Скачано: {ok}, ошибок: {err}. Индексация ПРОПУЩЕНА "
                             "(только парсинг) — запустите «Переиндексировать», когда нужно.\n")
                    fp.write(f"SUMMARY web_ok={ok} web_err={err} index_rc=skipped\n")
                    _set_all_progress({"phase": "parsed", "pct": 100})
            _web_job["log"] = _tail(logfile)
            _web_job["summary"] = _extract_summary(_web_job["log"])
            _web_job["ok"] = (err == 0 and rc == 0)
        except Exception as e:
            _web_job["ok"] = False
            _web_job["log"] = (_tail(logfile) + "\n" + str(e)).strip()
        _web_job["running"] = False
        _web_job["finished"] = time.time()
        try:
            db.ingest_log_save("Парсинг сайтов", _web_job.get("summary") or "",
                               _read_full_log(logfile))
        except Exception as e:
            print(f"[web] не удалось сохранить лог в БД: {e}")

    threading.Thread(target=run, daemon=True).start()
    tail = "затем — индексация" if index else "без индексации (только парсинг)"
    return {"ok": True, "msg": f"парсинг {len(urls)} сайт(ов) запущен; {tail}"}


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
    r = _bg(_job, "Индексация", [cmd], "/tmp/rag_index.log",
            save_label="Переиндексация с нуля" if reset else "Индексация")
    if r.get("ok"):
        r["msg"] = "индексация запущена"
    return r


def _stop_job(job: dict) -> dict:
    """Остановить фоновую задачу: прибить процесс и всю его группу."""
    import os as _os
    import signal as _sig
    if not job.get("running"):
        return {"ok": False, "msg": "задача не выполняется"}
    job["stopped"] = True
    proc = job.get("_proc")
    if proc is None:
        return {"ok": True, "msg": "остановка запрошена"}
    try:
        # прибиваем всю группу процессов (ingest.py + воркеры)
        try:
            _os.killpg(_os.getpgid(proc.pid), _sig.SIGTERM)
        except Exception:
            proc.terminate()
        for _ in range(20):
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        if proc.poll() is None:                      # не завершился — жёстко
            try:
                _os.killpg(_os.getpgid(proc.pid), _sig.SIGKILL)
            except Exception:
                proc.kill()
    except Exception as e:
        return {"ok": False, "msg": f"не удалось остановить: {e}"}
    return {"ok": True, "msg": "индексация останавливается"}


def stop_reindex() -> dict:
    """Остановить текущую индексацию/переиндексацию."""
    return _stop_job(_job)


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
        try:
            # пересоздаём пустую коллекцию активной векторной базы (Qdrant/Milvus),
            # чтобы чат не падал до переиндексации
            try:
                _dim = int(settings.get("EMBED_DIM"))
            except Exception:
                _dim = 1024
            vectorstore.ensure_collection(_dim, reset=True)
            # сбрасываем накопленное время обработки — оно относится к старому индексу
            _INGEST_STATS.unlink(missing_ok=True)
            try:
                import cache
                cache.bump("index")   # сброс кэша поиска/ответов
            except Exception:
                pass
            done.append(f"индекс ({vectorstore.backend()})")
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


def _in_docker() -> bool:
    """Приложение запущено в контейнере (Docker)."""
    if os.getenv("RAG_DOCKER"):
        return True
    try:
        return os.path.exists("/.dockerenv")
    except Exception:
        return False


def check_updates() -> dict:
    """Сравнить локальную версию с origin (git fetch). В Docker — иная схема обновления."""
    if _in_docker():
        return {"ok": True, "docker": True, "up_to_date": None,
                "msg": "Docker-вариант: обновление выполняется на хосте пересборкой образа. "
                       "Запустите на хосте update.cmd (он делает git pull и docker compose "
                       "up -d --build). Проверка/обновление из контейнера недоступны, т.к. код "
                       "вшит в образ, а приватный репозиторий из контейнера не тянется."}
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
    if _in_docker():
        return {"ok": False, "docker": True,
                "msg": "В Docker обновление из контейнера недоступно. На хосте запустите "
                       "update.cmd (git pull + пересборка образа: docker compose up -d --build)."}
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
    # Milvus: переводим простые must-match фильтры в нейтральные и считаем через фасад;
    # сложные (must_not/is_empty) поддерживаются приблизительно (facet по наличию поля).
    if vectorstore.is_milvus():
        try:
            must = (flt or {}).get("must") or []
            neutral = {}
            for c in must:
                if "key" in c and isinstance(c.get("match"), dict):
                    neutral[c["key"]] = c["match"].get("value")
            if neutral or not flt:
                return vectorstore.count(neutral or None)
            # must_not is_empty <key> → число строк, где поле задано
            mn = (flt or {}).get("must_not") or []
            for c in mn:
                k = ((c.get("is_empty") or {}).get("key"))
                if k:
                    return sum(h.get("count", 0) for h in vectorstore.facet(k, 100000))
        except Exception:
            pass
        return 0
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

    # ---- KAG (знание-усиленная генерация) ----
    kag = {
        "active": settings.get("ENGINE") == "kag",
        "backend": settings.get("LLM_BACKEND"),
        "model": settings.active_model(),
        "decompose": bool(settings.get("KAG_DECOMPOSE")),
        "max_hops": settings.get("KAG_MAX_HOPS"),
        "chunks_per_hop": settings.get("KAG_CHUNKS_PER_HOP"),
        "context_chunks": settings.get("KAG_CONTEXT_CHUNKS"),
        "mutual_index": bool(settings.get("KAG_MUTUAL_INDEX")),
        "use_graph": bool(settings.get("KAG_GRAPH")),
        "graph_mode": settings.get("KAG_GRAPH_MODE"),
        "graph_ready": bool(graph.get("ready")),
        "citations": bool(settings.get("KAG_REQUIRE_CITATIONS")),
        "temperature": settings.get("KAG_TEMPERATURE"),
        # текущие параметры поиска, которые использует мультихоп KAG
        "min_score": settings.get("MIN_SCORE"),
        "top_k_retrieve": settings.get("TOP_K_RETRIEVE"),
        "top_k_rerank": settings.get("TOP_K_RERANK"),
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

    # ---- Дополнительные коннекторы (для живой схемы работы) ----
    # Синонимы, справочник сотрудников и внешние API-хуки управляются через БД,
    # а не через конфиг, поэтому их состояние отдаём отдельным блоком.
    connectors: dict = {}
    try:
        import synonyms
        connectors["synonyms"] = {"enabled": bool(synonyms.enabled()),
                                  "count": len(db.syn_list())}
    except Exception:
        connectors["synonyms"] = {"enabled": False, "count": 0}
    try:
        connectors["org"] = db.org_meta()
    except Exception:
        connectors["org"] = {"count": 0}
    try:
        hooks = db.api_hooks_list()
        connectors["api_hooks"] = {
            "total": len(hooks),
            "enabled": sum(1 for h in hooks if h.get("enabled")),
        }
    except Exception:
        connectors["api_hooks"] = {"total": 0, "enabled": 0}
    try:
        import telegram_bot
        connectors["telegram"] = {
            "running": bool(telegram_bot._state.get("running")),
            "token_set": bool((settings.get("TELEGRAM_BOT_TOKEN") or "").strip()),
            "username": telegram_bot._state.get("username"),
            "error": telegram_bot._state.get("error"),
        }
    except Exception:
        connectors["telegram"] = {"running": False, "token_set": False}
    try:
        import sip_bridge
        import sip_phone
        as_st = getattr(sip_bridge, "_state", {})
        rg_st = getattr(sip_phone, "_state", {})
        try:
            rg_phase = sip_phone._phone_phase()      # фактическая фаза регистрации
        except Exception:
            rg_phase = "unknown"
        connectors["sip"] = {
            "audiosocket": {"enabled": bool(settings.get("SIP_ENABLED")),
                            "running": bool(as_st.get("running"))},
            "register": {"enabled": bool(settings.get("SIP_REGISTER_ENABLED")),
                         "registered": rg_phase == "registered",
                         "phase": rg_phase,
                         "running": bool(rg_st.get("running")),
                         "server": str(settings.get("SIP_SERVER") or "")},
            "active": int(as_st.get("active", 0)) + int(rg_st.get("active", 0)),
        }
    except Exception:
        connectors["sip"] = {}

    return {"qdrant": qd, "graph": graph, "finetune": ft,
            "hybrid": hybrid, "kag": kag, "usage": db.engine_usage(),
            "ingest": _ingest_summary(), "connectors": connectors,
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


def component_status(gpu: dict | None = None) -> dict:
    """Что где выполняется в реальном времени: устройство каждого компонента RAG
    (CPU/GPU/внешний), загружен ли он, память процесса (ОЗУ) и видеопамять (CUDA)."""
    MB = 1024 * 1024
    try:
        dev = settings.device()
    except Exception:
        dev = (settings.get("DEVICE") or "cpu").lower()
    g = gpu if gpu is not None else _gpu_info()
    devs = (g or {}).get("devices") or []
    gpu_name = devs[0].get("name") if devs else None

    def dl(d):
        if d == "cuda":
            return ("GPU: " + gpu_name) if gpu_name else "GPU (CUDA)"
        if d == "mps":
            return "GPU: Apple Metal (MPS)"
        return "CPU"

    # память процесса приложения: ОЗУ (RSS) и видеопамять (CUDA), выделенная этим процессом
    ram_mb = cuda_alloc = cuda_reserved = None
    try:
        import psutil
        ram_mb = round(psutil.Process().memory_info().rss / MB)
    except Exception:
        pass
    if dev == "cuda":
        try:
            import torch
            cuda_alloc = round(torch.cuda.memory_allocated() / MB)
            cuda_reserved = round(torch.cuda.memory_reserved() / MB)
        except Exception:
            pass

    comps = []
    # эмбеддинги и реранкер (torch, в этом процессе; @lru_cache → currsize>0 если загружены)
    try:
        import retriever
        comps.append({"name": "Эмбеддинги", "model": settings.get("EMBED_MODEL"),
                      "loaded": retriever._embedder.cache_info().currsize > 0, "where": dl(dev)})
        comps.append({"name": "Реранкер", "model": settings.get("RERANK_MODEL"),
                      "loaded": retriever._reranker.cache_info().currsize > 0, "where": dl(dev)})
    except Exception:
        pass
    # Whisper (STT) — грузится при первой транскрибации
    try:
        import loaders
        wback = (settings.get("WHISPER_BACKEND") or "faster").lower()
        wdev = (settings.get("DEVICE") or "cpu").lower()
        if wback == "mlx":
            comps.append({"name": "Whisper (STT)", "model": settings.get("WHISPER_MODEL"),
                          "loaded": None, "where": "GPU: Apple Metal (MPS)"})
        else:
            wloaded = getattr(loaders, "_FASTER_WHISPER", None) is not None
            comps.append({"name": "Whisper (STT)", "model": settings.get("WHISPER_MODEL"),
                          "loaded": wloaded, "where": dl(wdev if wdev in ("cuda", "mps") else "cpu"),
                          "note": "" if wloaded else "загрузится при первой транскрибации"})
    except Exception:
        pass
    # Генерация LLM и vision — внешний движок (Ollama/vLLM), считает на своём GPU
    backend = (settings.get("LLM_BACKEND") or "ollama").lower()
    if backend == "ollama":
        ourl = settings.get("OLLAMA_URL")
        if not ourl:
            try:
                import config as _c
                ourl = getattr(_c, "OLLAMA_URL", "http://localhost:11434")
            except Exception:
                ourl = "http://localhost:11434"
        llm_where = f"внешний: Ollama ({ourl}) — на GPU/CPU хоста, не видно из приложения"
    else:
        llm_where = "внешний: OpenAI-совместимый API (vLLM) — на GPU сервера"
    comps.append({"name": "Генерация LLM", "model": settings.active_model(),
                  "loaded": None, "where": llm_where, "external": True})
    vm = settings.get("VISION_MODEL")
    if vm:
        comps.append({"name": "Vision-модель", "model": vm, "loaded": None,
                      "where": llm_where, "external": True})

    return {
        "compute_device": dev, "device_label": dl(dev),
        "process": {"ram_mb": ram_mb, "cuda_alloc_mb": cuda_alloc,
                    "cuda_reserved_mb": cuda_reserved,
                    "gpu_name": gpu_name if dev == "cuda" else None},
        "components": comps,
        "gpus": [{"index": d.get("index"), "name": d.get("name"),
                  "mem_used": d.get("mem_used"), "mem_total": d.get("mem_total")} for d in devs],
    }


def _qdrant_stats() -> dict:
    """Быстрая сводка по коллекции Qdrant (для расширенной статистики)."""
    try:
        info = vectorstore.collection_info(backend_name="qdrant")
        return {"reachable": bool(info.get("exists")), "points": info.get("points_count"),
                "status": info.get("status")}
    except Exception as e:
        return {"reachable": False, "error": str(e)[:100]}


def _milvus_stats() -> dict:
    """Быстрая сводка по коллекции Milvus (для расширенной статистики). Безопасна,
    если pymilvus не установлен или Milvus недоступен — вернёт reachable=False."""
    try:
        if not vectorstore.ping("milvus"):
            return {"reachable": False}
        info = vectorstore.collection_info(backend_name="milvus")
        return {"reachable": True, "points": info.get("points_count"),
                "status": info.get("status")}
    except Exception as e:
        return {"reachable": False, "error": str(e)[:100]}


def _milvus_pkg_present() -> bool:
    """Установлен ли pymilvus (клиент Milvus) в окружении."""
    try:
        return _iu.find_spec("pymilvus") is not None
    except Exception:
        return False


def component_metrics() -> dict:
    """Расширенная статистика конвейера в реальном времени: по каждому компоненту —
    число обращений/ошибок и средняя задержка (из metrics), плюс ресурсы (точки
    Qdrant, размер БД, память Redis, устройство модели). Фронт из кумулятивных
    счётчиков считает скорость (запросов/с) между опросами."""
    import metrics
    snap = metrics.snapshot()
    C = snap.get("components", {})

    def cnt(name):
        d = C.get(name) or {}
        return {"calls": d.get("calls", 0), "errors": d.get("errors", 0),
                "avg_ms": d.get("avg_ms", 0.0)}

    try:
        dev = settings.device()
    except Exception:
        dev = (settings.get("DEVICE") or "cpu").lower()
    dev_label = {"cuda": "GPU (CUDA)", "mps": "GPU (Apple Metal)"}.get(dev, "CPU")

    try:
        import retriever
        emb_loaded = retriever._embedder.cache_info().currsize > 0
        rr_loaded = retriever._reranker.cache_info().currsize > 0
    except Exception:
        emb_loaded = rr_loaded = False

    active_db = "sqlite"
    try:
        active_db = db._dialect()
    except Exception:
        pass
    db_size = None
    try:
        if db.DB_PATH.exists():
            db_size = round(db.DB_PATH.stat().st_size / 1048576, 2)
    except Exception:
        pass

    engine = (settings.get("ENGINE") or "").lower()
    graph_on = bool(settings.get("GRAPH_RAG")) or engine == "lightrag" or bool(settings.get("KAG_GRAPH"))
    kag_on = engine == "kag"

    # LLM — из реестра llm_activity (генерация/vision/служебные), с токенами
    try:
        import llm_activity
        la = llm_activity.snapshot(1)
    except Exception:
        la = {}
    llm_calls = la.get("total_calls", 0)
    llm_gen_ms = la.get("total_gen_ms", 0)
    llm_avg_ms = round(llm_gen_ms / llm_calls, 1) if llm_calls else 0.0
    _tps = la.get("avg_tps")
    _ttok = la.get("total_tokens", 0)
    _lparts = [str(settings.active_model() or "")]
    if _tps:
        _lparts.append(f"{_tps} ток/с")
    if _ttok:
        _lparts.append(f"токенов {_ttok}")
    if la.get("running"):
        _lparts.append(f"идёт {la['running']}")

    q = _qdrant_stats()
    _vbackend = vectorstore.backend()
    m = _milvus_stats() if (_vbackend == "milvus" or _milvus_pkg_present()) else {"reachable": False}
    # лёгкая проверка Redis без сканирования ключей (status() сканирует — тяжело для частого опроса)
    rstat = {"enabled": False, "reachable": False}
    try:
        import cache
        rstat["enabled"] = cache.enabled()
        cc = cache.client()
        if cc is not None:
            info = cc.info()
            hits = info.get("keyspace_hits") or 0
            misses = info.get("keyspace_misses") or 0
            tot = hits + misses
            rstat.update(reachable=True, used_memory=info.get("used_memory_human"),
                         total_keys=cc.dbsize(), hits=hits, misses=misses,
                         hit_rate=round(hits / tot * 100, 1) if tot else None)
    except Exception:
        pass

    comps = [
        {"key": "qdrant", "name": "Qdrant", "group": "Хранилище",
         "desc": "Векторная БД: хранит эмбеддинги чанков и выполняет ANN-поиск (dense) "
                 "при каждом вопросе." + (" Активный бэкенд." if _vbackend == "qdrant"
                 else " Сейчас неактивен (активен Milvus)."),
         **(cnt("qdrant") if _vbackend == "qdrant" else {"calls": 0, "errors": 0, "avg_ms": 0.0}),
         "resource": {"reachable": q.get("reachable"),
                      "label": ("✅ активна · " if _vbackend == "qdrant" else "")
                               + (f"точек: {q.get('points')}" if q.get("points") is not None else "—")
                               + (f" · {q.get('status')}" if q.get("status") else ""),
                      "points": q.get("points"), "status": q.get("status"),
                      "active": _vbackend == "qdrant"}},
        {"key": "milvus", "name": "Milvus", "group": "Хранилище",
         "desc": "Альтернативная векторная СУБД (масштаб, GPU-индексы). "
                 + ("Активный бэкенд." if _vbackend == "milvus" else
                    "Установите и перенесите данные в блоке «Векторная база: Qdrant ⇄ Milvus»."),
         **(cnt("qdrant") if _vbackend == "milvus" else {"calls": 0, "errors": 0, "avg_ms": 0.0}),
         "resource": {"reachable": bool(m.get("reachable")),
                      "label": ("✅ активна · " if _vbackend == "milvus" else "")
                               + (f"точек: {m.get('points')}" if m.get("points") is not None
                                  else ("доступен" if m.get("reachable") else "не установлен/выключен")),
                      "points": m.get("points"), "status": m.get("status"),
                      "active": _vbackend == "milvus"}},
        {"key": "embed", "name": "Эмбеддер (bge-m3)", "group": "Модели",
         "desc": "Превращает текст вопроса/чанков в векторы. Выполняется в процессе "
                 "приложения на устройстве DEVICE.", **cnt("embed"),
         "resource": {"reachable": emb_loaded, "label": dev_label + (" · загружен" if emb_loaded else " · не загружен"),
                      "device": dev, "model": settings.get("EMBED_MODEL")}},
        {"key": "rerank", "name": "Реранкер (bge-reranker)", "group": "Модели",
         "desc": "Cross-encoder: переоценивает релевантность кандидатов к вопросу. "
                 "Обычно самый тяжёлый по времени этап поиска.", **cnt("rerank"),
         "resource": {"reachable": rr_loaded, "label": dev_label + (" · загружен" if rr_loaded else " · не загружен"),
                      "device": dev, "model": settings.get("RERANK_MODEL")}},
        {"key": "llm", "name": "Генерация LLM", "group": "LLM",
         "desc": "Формирование ответов моделью (чат/Телеграм/VoIP), описание изображений "
                 "vision-моделью и служебные вызовы (фильтр запроса, ИИ-интент). Считает внешний "
                 "движок (Ollama/vLLM); скорость — токены/с.",
         "calls": llm_calls, "errors": la.get("total_errors", 0), "avg_ms": llm_avg_ms,
         "resource": {"reachable": llm_calls > 0 or bool(la.get("running")),
                      "label": " · ".join([p for p in _lparts if p]),
                      "tps": _tps, "total_tokens": _ttok}},
        {"key": "lightrag", "name": "LightRAG (граф-RAG)", "group": "Движки",
         "desc": "Ответы по графу знаний для сводных/глобальных вопросов. Используется, "
                 "когда включён граф-режим.", **cnt("lightrag"),
         "resource": {"reachable": graph_on, "label": "включён" if graph_on else "выключен"}},
        {"key": "kag", "name": "KAG", "group": "Движки",
         "desc": "Knowledge-Augmented Generation: многошаговая декомпозиция вопроса и "
                 "обход графа. Активен при ENGINE=kag.", **cnt("kag"),
         "resource": {"reachable": kag_on, "label": "включён" if kag_on else "выключен"}},
        {"key": "sqlite", "name": "SQLite", "group": "База данных",
         "desc": "Локальная БД: журнал, настройки, метаданные, оценки, кэш-версии. "
                 "Активна, если не настроен внешний сервер БД.",
         **(cnt("db:sqlite") if active_db == "sqlite" else {"calls": 0, "errors": 0, "avg_ms": 0.0}),
         "resource": {"reachable": active_db == "sqlite",
                      "label": (f"активна · {db_size} МБ" if active_db == "sqlite" and db_size is not None
                                else ("активна" if active_db == "sqlite" else "не используется")),
                      "size_mb": db_size if active_db == "sqlite" else None}},
        {"key": "postgresql", "name": "PostgreSQL", "group": "База данных",
         "desc": "Внешний сервер БД (альтернатива SQLite): те же журнал/настройки/каталог "
                 "документов. Активен, если выбран в настройках БД.",
         **(cnt("db:postgresql") if active_db == "postgresql" else {"calls": 0, "errors": 0, "avg_ms": 0.0}),
         "resource": {"reachable": active_db == "postgresql",
                      "label": "активна" if active_db == "postgresql" else "не используется"}},
        {"key": "mysql", "name": "MySQL", "group": "База данных",
         "desc": "Внешний сервер БД (альтернатива SQLite/PostgreSQL): те же журнал/настройки/"
                 "каталог документов. Активен, если выбран в настройках БД.",
         **(cnt("db:mysql") if active_db == "mysql" else {"calls": 0, "errors": 0, "avg_ms": 0.0}),
         "resource": {"reachable": active_db == "mysql",
                      "label": "активна" if active_db == "mysql" else "не используется"}},
        {"key": "redis", "name": "Redis", "group": "Кэш",
         "desc": "Опциональный кэш агрегатов и общий межпроцессный реестр. Ускоряет "
                 "статистику/поиск; при отсутствии всё работает напрямую.",
         "calls": rstat.get("hits", 0) if rstat.get("reachable") else 0,
         "errors": rstat.get("misses", 0) if rstat.get("reachable") else 0, "avg_ms": 0.0,
         "resource": {"reachable": bool(rstat.get("reachable")),
                      "label": (f"{rstat.get('used_memory','?')} · ключей {rstat.get('total_keys','?')}"
                                if rstat.get("reachable")
                                else ("включён, недоступен" if rstat.get("enabled") else "выключен")),
                      "enabled": rstat.get("enabled"), "hit_rate": rstat.get("hit_rate"),
                      "used_memory": rstat.get("used_memory")}},
    ]
    return {"ts": snap.get("ts"), "components": comps}


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
    # фактическое устройство вычислений (эмбеддинги/реранк) — с учётом доступности
    # torch/CUDA/MPS. Считается только здесь (в процессе приложения, где torch уже
    # загружен), а не в _gpu_info(), который дёргает и лёгкий монитор-подпроцесс.
    try:
        out["gpu"]["compute_device"] = settings.device()
    except Exception:
        out["gpu"]["compute_device"] = (settings.get("DEVICE") or "cpu").lower()

    # Fallback: если системная утилита (nvidia-smi) недоступна — частый случай в
    # Docker-контейнере, — но CUDA видна приложению (GPU проброшен), покажем карточки
    # GPU через torch: имя и видеопамять по каждой карте (утилизацию torch не даёт).
    if out["gpu"].get("vendor") == "none":
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                devs = []
                for i in range(torch.cuda.device_count()):
                    used = total = mutil = None
                    try:
                        free, tot = torch.cuda.mem_get_info(i)
                        used = (tot - free) // (1024 * 1024)
                        total = tot // (1024 * 1024)
                        mutil = round(used * 100 / total) if total else None
                    except Exception:
                        pass
                    devs.append({"index": str(i), "name": torch.cuda.get_device_name(i),
                                 "mem_used": used, "mem_total": total, "mem_util": mutil,
                                 "util": None, "temp": None, "power": None,
                                 "power_limit": None, "fan": None, "via": "torch"})
                if devs:
                    out["gpu"] = {"vendor": "cuda", "devices": devs,
                                  "compute_device": out["gpu"].get("compute_device")}
        except Exception:
            pass

    # что где выполняется (CPU/GPU/внешний) + память процесса — в реальном времени
    try:
        out["components"] = component_status(out.get("gpu"))
    except Exception as e:
        out["components"] = {"error": str(e)}
    return out


_PERIOD_RU = {"hour": "час", "day": "день", "week": "неделя",
              "month": "месяц", "year": "год"}


def _hw_context() -> dict:
    """Контекст железа для рекомендаций: ядра CPU, объём ОЗУ, GPU, бэкенд LLM."""
    ctx = {"cpu_cores": None, "mem_total_gb": None, "gpu_vendor": "none",
           "gpu_count": 0, "gpu_mem_total_gb": None, "backend": None, "device": None}
    try:
        import psutil
        ctx["cpu_cores"] = psutil.cpu_count(logical=True)
        ctx["mem_total_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        ctx["cpu_cores"] = os.cpu_count()
    try:
        g = _gpu_info()
        ctx["gpu_vendor"] = g.get("vendor", "none")
        devs = g.get("devices") or []
        ctx["gpu_count"] = len(devs) if ctx["gpu_vendor"] in ("nvidia", "amd") else 0
        mts = [d.get("mem_total") for d in devs if d.get("mem_total")]
        if mts:
            ctx["gpu_mem_total_gb"] = round(max(mts) / 1024, 1)   # МБ → ГБ
    except Exception:
        pass
    try:
        ctx["backend"] = settings.get("LLM_BACKEND")
        ctx["device"] = settings.get("DEVICE")
    except Exception:
        pass
    return ctx


def _best_period(periods: dict) -> tuple[str, dict]:
    """Самое длинное окно с достаточным числом выборок (>=30), иначе самое полное."""
    order = ["year", "month", "week", "day", "hour"]
    for name in order:
        p = periods.get(name) or {}
        if p.get("samples", 0) >= 30:
            return name, p
    # фолбэк — окно с максимумом выборок
    best = max(periods.items(), key=lambda kv: kv[1].get("samples", 0),
               default=("day", {}))
    return best[0], best[1]


def _hw_recommendations(periods: dict, ctx: dict) -> list[dict]:
    """Рекомендации по железу на основе агрегатов загрузки. Уровни:
    critical (срочно), warn (внимание), ok (запас), info (справка)."""
    recs: list[dict] = []
    pname, p = _best_period(periods)
    win = _PERIOD_RU.get(pname, pname)
    n = p.get("samples", 0)
    if n < 10:
        recs.append({"level": "info",
                     "text": "Недостаточно данных для рекомендаций — статистика "
                             "загрузки накапливается, зайдите позже."})
        return recs

    def g(k):
        return p.get(k)

    # ---- CPU ----
    ca, cm = g("cpu_avg"), g("cpu_max")
    cores = ctx.get("cpu_cores")
    cores_s = f", ядер: {cores}" if cores else ""
    if cm is not None and cm >= 95 and (ca or 0) >= 60:
        recs.append({"level": "critical",
                     "text": f"CPU постоянно перегружен (средн. {ca}%, пик {cm}% за {win}"
                             f"{cores_s}). Возьмите процессор с большим числом ядер/частотой "
                             "или вынесите эмбеддинг и LLM на отдельный сервер/GPU."})
    elif cm is not None and cm >= 90:
        recs.append({"level": "warn",
                     "text": f"Бывают пики загрузки CPU до {cm}% (средн. {ca}% за {win}). "
                             "Под пиковой индексацией поможет более мощный CPU; следите за "
                             "временем ответа."})
    elif ca is not None and ca < 20:
        recs.append({"level": "ok",
                     "text": f"CPU с большим запасом (средн. {ca}%, пик {cm}% за {win}) — "
                             "апгрейд процессора не требуется."})

    # ---- Память ----
    ma, mm, sw = g("mem_avg"), g("mem_max"), g("swap_max")
    ram_s = f", всего ОЗУ: {ctx['mem_total_gb']} ГБ" if ctx.get("mem_total_gb") else ""
    if (mm is not None and mm >= 92) or (sw is not None and sw >= 25):
        recs.append({"level": "critical",
                     "text": f"Память на пределе (пик {mm}%, swap до {sw}% за {win}{ram_s}). "
                             "Добавьте ОЗУ — нехватка вызывает своппинг и резко замедляет "
                             "ответы; ориентир +50–100% к текущему объёму."})
    elif mm is not None and mm >= 80:
        recs.append({"level": "warn",
                     "text": f"Память используется плотно (пик {mm}%, средн. {ma}% за {win}"
                             f"{ram_s}). Стоит запланировать увеличение ОЗУ."})
    elif ma is not None and ma < 40:
        recs.append({"level": "ok",
                     "text": f"Памяти достаточно (средн. {ma}%, пик {mm}% за {win}{ram_s})."})

    # ---- GPU ----
    has_gpu = ctx.get("gpu_vendor") in ("nvidia", "amd")
    gmm, ga = g("gpu_mem_max"), g("gpu_avg")
    vram_s = (f", VRAM: {ctx['gpu_mem_total_gb']} ГБ" if ctx.get("gpu_mem_total_gb") else "")
    if has_gpu:
        if gmm is not None and gmm >= 92:
            recs.append({"level": "critical",
                         "text": f"Видеопамять почти исчерпана (пик {gmm}% за {win}{vram_s}). "
                                 "Возьмите GPU с большим объёмом VRAM, используйте меньшую/"
                                 "квантованную модель (AWQ/Int4) или снизьте VLLM_MAX_LEN / "
                                 "размер контекста."})
        elif gmm is not None and gmm >= 80:
            recs.append({"level": "warn",
                         "text": f"Видеопамять заполняется (пик {gmm}% за {win}{vram_s}). "
                                 "Следите за запасом при росте модели/контекста."})
        elif ga is not None and ga < 15:
            recs.append({"level": "ok",
                         "text": f"GPU слабо загружен (средн. {ga}% за {win}) — есть запас "
                                 "под более крупную модель или больший батч."})
    else:
        backend = (ctx.get("backend") or "").lower()
        dev = (ctx.get("device") or "").lower()
        if backend in ("vllm", "openai") or dev == "cuda":
            recs.append({"level": "warn",
                         "text": "Выбран GPU-бэкенд генерации (vLLM/CUDA), но видеокарта не "
                                 "обнаружена. Для ускорения нужна NVIDIA GPU; иначе генерация "
                                 "идёт на CPU и медленнее."})
        elif (ca is not None and ca >= 60):
            recs.append({"level": "info",
                         "text": "GPU не обнаружен, а CPU нагружен. Видеокарта NVIDIA заметно "
                                 "ускорит эмбеддинг, реранкинг и работу LLM."})

    # ---- Диск ----
    dm = g("disk_max")
    if dm is not None and dm >= 90:
        recs.append({"level": "critical",
                     "text": f"Диск почти заполнен (пик {dm}% за {win}). Расширьте хранилище "
                             "или очистите данные — нехватка места ломает индексацию и БД."})
    elif dm is not None and dm >= 80:
        recs.append({"level": "warn",
                     "text": f"На диске остаётся мало места (пик {dm}% за {win}). "
                             "Запланируйте расширение."})

    if not recs:
        recs.append({"level": "ok",
                     "text": f"За период «{win}» ресурсы в норме — текущей конфигурации "
                             "достаточно, апгрейд не требуется."})
    return recs


def server_history() -> dict:
    """История загрузки по окнам + рекомендации по железу для раздела
    «Загрузка сервера»."""
    periods = db.server_load_stats()
    ctx = _hw_context()
    recs = _hw_recommendations(periods, ctx)
    total = sum(p.get("samples", 0) for p in periods.values())
    since = None
    yr = periods.get("year") or {}
    if yr.get("since"):
        since = yr["since"]
    return {
        "periods": periods,
        "period_labels": _PERIOD_RU,
        "recommendations": recs,
        "hardware": ctx,
        "samples_total": (periods.get("year") or {}).get("samples", 0),
        "since": since,
        "monitoring": _monitor_running(),
    }


def _monitor_running() -> bool:
    try:
        import monitor
        return monitor.running()
    except Exception:
        return False


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
        "ingest_breakdown": _ingest_breakdown(),
    }

    kag = {
        "active": settings.get("ENGINE") == "kag",
        "decompose": bool(settings.get("KAG_DECOMPOSE")),
        "max_hops": settings.get("KAG_MAX_HOPS"),
        "chunks_per_hop": settings.get("KAG_CHUNKS_PER_HOP"),
        "context_chunks": settings.get("KAG_CONTEXT_CHUNKS"),
        "mutual_index": bool(settings.get("KAG_MUTUAL_INDEX")),
        "use_graph": bool(settings.get("KAG_GRAPH")),
        "graph_mode": settings.get("KAG_GRAPH_MODE"),
        "graph_ready": bool(graph.get("ready")),
        "citations": bool(settings.get("KAG_REQUIRE_CITATIONS")),
        "temperature": settings.get("KAG_TEMPERATURE"),
        "backend": settings.get("LLM_BACKEND"),
        "model": settings.active_model(),
    }

    return {"qdrant": qd, "graph": graph, "finetune": ft, "kag": kag,
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


def _ingest_breakdown() -> dict:
    """Разбивка времени индексации (парсинг vs эмбеддинг) из ingest_stats.json:
    суммарно, по типам файлов и по категориям чанков. Старые записи без раздельных
    таймингов учитываются в «парсинге» (поле ms)."""
    files = (_ingest_stats().get("files") or {})
    total_parse = total_embed = total_chunks = 0
    by_ftype: dict = {}
    by_cat: dict = {}
    for v in files.values():
        if not isinstance(v, dict):
            continue
        p = v.get("parse_ms")
        e = v.get("embed_ms")
        if p is None and e is None:        # старый формат — только суммарный ms
            p, e = v.get("ms", 0), 0
        p = int(p or 0)
        e = int(e or 0)
        ch = int(v.get("chunks", 0) or 0)
        ft = (v.get("ftype") or "—")
        cat = (v.get("category") or "—")
        total_parse += p
        total_embed += e
        total_chunks += ch
        a = by_ftype.setdefault(ft, {"parse_ms": 0, "embed_ms": 0, "chunks": 0, "files": 0})
        a["parse_ms"] += p; a["embed_ms"] += e; a["chunks"] += ch; a["files"] += 1
        b = by_cat.setdefault(cat, {"parse_ms": 0, "embed_ms": 0, "chunks": 0, "files": 0})
        b["parse_ms"] += p; b["embed_ms"] += e; b["chunks"] += ch; b["files"] += 1
    return {
        "overall": {"parse_ms": total_parse, "embed_ms": total_embed,
                    "chunks": total_chunks, "files": len(files)},
        "by_ftype": by_ftype, "by_category": by_cat,
    }


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
            for h in vectorstore.facet("source", 100000):
                out[h.get("value")] = h.get("count", 0)
        except Exception:
            pass
        return out

    # источники, у которых есть чанк-описание vision-модели (payload vision_desc=true)
    def _facet_described():
        out = set()
        try:
            for h in vectorstore.facet("source", 100000, flt={"vision_desc": True}):
                if h.get("count", 0) > 0 and h.get("value"):
                    out.add(h.get("value"))
        except Exception:
            pass
        return out

    try:
        import cache
        counts = cache.get_or_set("facet:" + str(settings.get("QDRANT_COLLECTION")),
                                  60, _facet, ns="index")
        described = cache.get_or_set("facet_desc:" + str(settings.get("QDRANT_COLLECTION")),
                                     60, _facet_described, ns="index")
    except Exception:
        counts = _facet()
        described = _facet_described()

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
    described = described or set()
    described_count = sum(1 for f in files if f["path"] in described)
    for f in files:                       # пометка для строки таблицы
        f["described"] = f["path"] in described
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
            "described_count": described_count,
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
    # маркер чанков-описаний, добавленных vision-моделью (см. loaders._describe_image_part)
    desc_mark = "Описание изображения (vision-модель)"
    # источник — PostgreSQL: отдаём сохранённый текст из doc_catalog
    if _catalog_pg_active():
        r = db.catalog_text(source, max_chars)
        if r is None:
            return {"ok": True, "source": source, "text": "", "chunks": 0,
                    "method": _file_method(Path(source).suffix),
                    "note": "файл отсутствует в каталоге PostgreSQL"}
        text, desc = r["text"], ""
        if desc_mark in (text or ""):     # отделяем описание LLM от остального текста
            segs = text.split(desc_mark)
            text = segs[0].strip()
            desc = "\n\n".join(s.lstrip(" :\n").strip() for s in segs[1:] if s.strip())
        llm_meta = {k: r[k] for k in ("product", "topic", "doc_type", "doc_category")
                    if isinstance(r.get(k), str) and r[k].strip()}
        return {"ok": True, "source": source, "text": text, "description": desc,
                "llm_meta": llm_meta,
                "chunks": None, "method": r.get("method") or _file_method(Path(source).suffix),
                "n_chars": r.get("n_chars"), "truncated": r.get("truncated"),
                "from": "postgresql"}
    points, next_off = [], None
    try:
        for _ in range(40):  # до ~10k чанков на файл
            pts, next_off = vectorstore.scroll(flt={"source": source}, limit=256,
                                               offset=next_off, with_payload=True,
                                               with_vectors=False)
            points.extend(pts)
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

    # LLM-аннотация: структурированные метаданные, извлечённые LLM при индексации
    # (enrich.extract_structured → payload product/topic/doc_type/doc_category, при
    # включённой настройке LLM_METADATA). Одинаковы для всех чанков файла.
    llm_meta = {}
    for p in points:
        pl = p.get("payload") or {}
        for k in ("product", "topic", "doc_type", "doc_category"):
            v = pl.get(k)
            if k not in llm_meta and isinstance(v, str) and v.strip():
                llm_meta[k] = v.strip()
        if len(llm_meta) >= 4:
            break

    # отделяем чанки-описания vision-модели от извлечённого текста (OCR/инструменты)
    desc_parts, parts, total, truncated = [], [], 0, False
    for p in points:
        t = ((p.get("payload") or {}).get("text") or "").strip()
        if not t:
            continue
        if t.startswith(desc_mark):
            body = t.split("\n", 1)[1].strip() if "\n" in t else ""
            if body:
                desc_parts.append(body)
            continue
        if total + len(t) > max_chars:
            parts.append(t[:max(0, max_chars - total)])
            truncated = True
            break
        parts.append(t)
        total += len(t) + 2
    return {"ok": True, "source": source, "text": "\n\n".join(parts),
            "description": "\n\n".join(desc_parts), "llm_meta": llm_meta,
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


def oda_install() -> dict:
    """Установить/проверить ODA File Converter (запасной конвертер DWG→DXF).
    Авто-установка возможна не всегда (ODA требует ручной загрузки с сайта после
    регистрации) — тогда возвращаем ссылку и инструкцию."""
    import platform
    import shutil as _sh
    import loaders
    log: list[str] = []

    def run(cmd):
        log.append("$ " + " ".join(cmd))
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            out = (p.stdout or "") + (p.stderr or "")
            if out.strip():
                log.append(out.strip()[:4000])
            return p.returncode
        except Exception as e:
            log.append(f"[ошибка запуска] {e}")
            return 1

    cur = loaders.find_oda_converter()
    if cur:
        return {"ok": True, "msg": "ODA File Converter уже установлен", "path": cur,
                "log": f"Найден: {cur}"}

    link = "https://www.opendesign.com/guestfiles/oda_file_converter"
    sysname = platform.system().lower()
    if "darwin" in sysname and _sh.which("brew"):
        run(["brew", "install", "--cask", "oda-file-converter"])
        cur = loaders.find_oda_converter()
        if cur:
            return {"ok": True, "msg": "установлен через Homebrew", "path": cur,
                    "log": "\n".join(log)}
    elif _sh.which("apt-get"):
        # ставим из локального дистрибутива vendor/oda/*.deb + xvfb (общий скрипт)
        script = str(ROOT / "scripts" / "install_oda.sh")
        if os.path.exists(script):
            run(["bash", script, str(ROOT)])
        else:
            run(["apt-get", "install", "-y", "xvfb"])
        cur = loaders.find_oda_converter()
        if cur:
            return {"ok": True,
                    "msg": "ODA File Converter установлен из локального дистрибутива (vendor/oda)",
                    "path": cur, "log": "\n".join(log)}

    return {
        "ok": False,
        "msg": "Автоустановка не удалась: положите дистрибутив ODA (.deb для Linux) в папку "
               "vendor/oda/ репозитория и повторите, либо скачайте вручную с сайта ODA.",
        "link": link,
        "log": "\n".join(log) + (
            f"\n\nСкачайте ODA File Converter: {link}\n"
            "• macOS: откройте .dmg и перетащите приложение в /Applications — путь "
            "определится автоматически.\n"
            "• Linux: установите .deb/.rpm/.run; для сервера без дисплея нужен пакет "
            "xvfb (устанавливается этой кнопкой на apt-системах).\n"
            "• Если установили в нестандартное место — укажите путь к исполняемому файлу "
            "в настройке ODA_CONVERTER_PATH.\n"
            "После установки нажмите кнопку ещё раз для проверки."),
    }


def _redis_ping(host: str = "127.0.0.1", port: int = 6379) -> bool:
    """Проверить доступность Redis без зависимости от настроек (для установки)."""
    import socket as _s
    try:
        with _s.create_connection((host, port), timeout=2) as c:
            c.sendall(b"PING\r\n")
            return b"PONG" in c.recv(64)
    except Exception:
        return False


# ==================== Milvus: установка, миграция, переключение ====================

_milvus_job: dict = {"running": False, "phase": "", "direction": "", "done": 0,
                     "total": 0, "pct": 0, "error": None, "finished_ts": None,
                     "log": []}
_milvus_lock = threading.Lock()


def _mlog(msg: str) -> None:
    line = time.strftime("%H:%M:%S ") + msg
    _milvus_job["log"].append(line)
    _milvus_job["log"] = _milvus_job["log"][-200:]
    print("[milvus] " + msg, flush=True)


def milvus_status() -> dict:
    """Состояние Milvus и Qdrant для UI: установлен ли клиент, режим, доступность,
    число точек в каждом хранилище, активный бэкенд, ход миграции."""
    active = vectorstore.backend()
    pkg = _milvus_pkg_present()
    mode = (settings.get("MILVUS_MODE") or "lite").lower()
    out = {
        "active": active,
        "pkg_present": pkg,
        "docker": _in_docker(),
        "mode": mode,
        "index_type": settings.get("MILVUS_INDEX_TYPE"),
        "uri": "",
        "milvus": {"reachable": False, "points": None},
        "qdrant": _qdrant_stats(),
        "job": dict(_milvus_job),
    }
    try:
        out["uri"] = vectorstore.milvus_uri()
    except Exception:
        pass
    if pkg:
        out["milvus"] = _milvus_stats()
    return out


def milvus_install(mode: str | None = None, index_type: str | None = None) -> dict:
    """Установить клиент Milvus (pip pymilvus) и зафиксировать режим/индекс в настройках.

    - режим lite: одного pip install достаточно — встроенный Milvus работает в процессе;
    - режим standalone: клиент тот же, но нужен внешний сервер Milvus (контейнеры
      milvus+etcd+minio). В Docker их поднимает docker-compose (профиль milvus) —
      подсказываем командой; сам образ приложения уже содержит pymilvus после сборки.
    НЕ переключает активный бэкенд (это делается отдельно, после проверки миграции)."""
    log: list[str] = []

    def run(cmd, **kw):
        log.append("$ " + " ".join(cmd))
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, **kw)
            out = (p.stdout or "") + (p.stderr or "")
            if out.strip():
                log.append(out.strip()[-4000:])
            return p.returncode
        except Exception as e:
            log.append(f"[ошибка запуска] {e}")
            return 1

    changes: dict = {}
    if mode in ("lite", "standalone"):
        changes["MILVUS_MODE"] = mode
    if index_type:
        changes["MILVUS_INDEX_TYPE"] = index_type
    if changes:
        settings.update(changes)
    mode = (settings.get("MILVUS_MODE") or "lite").lower()

    # 1) клиент pymilvus (+ milvus-lite для встроенного режима на Linux/Mac)
    if not _milvus_pkg_present():
        log.append("Устанавливаю pymilvus …")
        rc = run([sys.executable, "-m", "pip", "install", "--break-system-packages",
                  "-U", "pymilvus>=2.4"])
        if rc != 0:
            run([sys.executable, "-m", "pip", "install", "-U", "pymilvus>=2.4"])
    else:
        log.append("pymilvus уже установлен.")

    # перезагрузим модуль pymilvus видимость (find_spec кэша нет — новый процесс не нужен)
    if not _milvus_pkg_present():
        return {"ok": False, "msg": "не удалось установить pymilvus (см. лог)",
                "log": "\n".join(log)}

    vectorstore.milvus_reset_client()

    if mode == "standalone":
        uri = ""
        try:
            uri = vectorstore.milvus_uri()
        except Exception:
            pass
        reachable = vectorstore.ping("milvus")
        if reachable:
            return {"ok": True,
                    "msg": f"pymilvus установлен; сервер Milvus доступен ({uri})",
                    "log": "\n".join(log + [f"Ping Milvus OK: {uri}"])}
        hint = ("Клиент установлен, но сервер Milvus (standalone) не отвечает. "
                "Поднимите сервисы Milvus: ")
        if _in_docker():
            hint += ("на хосте выполните `milvus.cmd` (или "
                     "`docker compose --profile milvus up -d` — сервисы etcd/minio/milvus "
                     "есть в compose), затем укажите MILVUS_URI=http://milvus:19530.")
        else:
            hint += ("запустите Milvus Standalone (docker-compose Milvus) и укажите "
                     "MILVUS_URI, например http://127.0.0.1:19530.")
        return {"ok": False, "msg": hint, "log": "\n".join(log + [f"URI: {uri}"])}

    # lite: проверяем, что встроенный Milvus поднимается и коллекция создаётся
    try:
        dim = int(settings.get("EMBED_DIM"))
        vectorstore.ensure_collection(dim, reset=False, backend_name="milvus")
        ok = vectorstore.ping("milvus")
        info = vectorstore.collection_info(backend_name="milvus")
        return {"ok": bool(ok),
                "msg": (f"Milvus Lite готов: {vectorstore.milvus_uri()} · "
                        f"точек {info.get('points_count', 0)}") if ok else
                       "pymilvus установлен, но встроенный Milvus не поднялся (см. лог)",
                "log": "\n".join(log + [f"URI: {vectorstore.milvus_uri()}"])}
    except Exception as e:
        return {"ok": False,
                "msg": f"pymilvus установлен, но инициализация Milvus Lite не удалась: {e}",
                "log": "\n".join(log + [str(e)])}


def _migrate_worker(direction: str) -> None:
    """Фоновая миграция векторов между хранилищами (src→dst) с прогрессом."""
    src, dst = ("qdrant", "milvus") if direction == "to_milvus" else ("milvus", "qdrant")
    try:
        dim = int(settings.get("EMBED_DIM"))
        _mlog(f"Миграция {src} → {dst} начата.")
        try:
            total = int(vectorstore.collection_info(backend_name=src).get("points_count") or 0)
        except Exception:
            total = 0
        with _milvus_lock:
            _milvus_job.update(total=total, done=0, pct=2, phase="prepare")
        # целевую коллекцию создаём заново (полная перезапись)
        vectorstore.ensure_collection(dim, reset=True, backend_name=dst)
        _mlog(f"Целевая коллекция ({dst}) пересоздана. Точек к переносу: {total}.")
        with _milvus_lock:
            _milvus_job.update(pct=5, phase="copy")

        batch: list = []
        BATCH = 512
        done = 0

        def _flush():
            nonlocal batch
            if not batch:
                return
            vectorstore.upsert_to(dst, batch, dim=dim)
            batch = []

        for p in vectorstore.iterate_all(batch=1000, backend_name=src):
            vec = p.get("vector")
            if not vec:
                continue
            batch.append({"id": p.get("id") or "", "vector": vec,
                          "payload": p.get("payload") or {}})
            if len(batch) >= BATCH:
                _flush()
            done += 1
            if total and done % 500 == 0:
                pct = 5 + int(done * 90 / max(total, 1))
                with _milvus_lock:
                    _milvus_job.update(done=done, pct=min(95, pct))
                _mlog(f"Перенесено {done}/{total} …")
        _flush()

        # сверка
        try:
            dst_n = int(vectorstore.collection_info(backend_name=dst).get("points_count") or 0)
        except Exception:
            dst_n = done
        _mlog(f"Готово. Источник: {total}, перенесено: {done}, в приёмнике: {dst_n}.")
        with _milvus_lock:
            _milvus_job.update(running=False, done=done, pct=100, phase="done",
                               finished_ts=time.time(),
                               error=None if (not total or dst_n >= min(done, total)) else
                               f"расхождение: приёмник {dst_n} из {total}")
    except Exception as e:
        _mlog(f"ОШИБКА миграции: {e}")
        with _milvus_lock:
            _milvus_job.update(running=False, pct=0, phase="error",
                               error=str(e), finished_ts=time.time())


def milvus_migrate(direction: str) -> dict:
    """Запустить полную миграцию векторов. direction: to_milvus | to_qdrant.
    Не переключает активный бэкенд — переключение делается кнопкой отдельно."""
    if direction not in ("to_milvus", "to_qdrant"):
        return {"ok": False, "msg": "неизвестное направление миграции"}
    if _milvus_job.get("running"):
        return {"ok": False, "msg": "миграция уже выполняется"}
    if direction == "to_milvus" and not _milvus_pkg_present():
        return {"ok": False, "msg": "сначала установите Milvus (кнопка «Установить Milvus»)"}
    # проверим доступность обоих хранилищ
    src, dst = ("qdrant", "milvus") if direction == "to_milvus" else ("milvus", "qdrant")
    if not vectorstore.ping(src):
        return {"ok": False, "msg": f"источник ({src}) недоступен"}
    if not vectorstore.ping(dst):
        return {"ok": False, "msg": f"приёмник ({dst}) недоступен — установите/поднимите его"}
    with _milvus_lock:
        _milvus_job.update(running=True, direction=direction, phase="start", done=0,
                           total=0, pct=1, error=None, finished_ts=None, log=[])
    threading.Thread(target=_migrate_worker, args=(direction,), daemon=True).start()
    return {"ok": True, "msg": f"Миграция {direction} запущена", "job": dict(_milvus_job)}


def milvus_switch(target: str) -> dict:
    """Переключить активную векторную базу (qdrant|milvus). Требует перезапуска сервиса,
    т.к. поисковые/индексирующие модули читают бэкенд при старте."""
    target = (target or "").strip().lower()
    if target not in ("qdrant", "milvus"):
        return {"ok": False, "msg": "цель должна быть qdrant или milvus"}
    if target == "milvus":
        if not _milvus_pkg_present():
            return {"ok": False, "msg": "Milvus не установлен"}
        if not vectorstore.ping("milvus"):
            return {"ok": False, "msg": "Milvus недоступен — не переключаю"}
    else:
        if not vectorstore.ping("qdrant"):
            return {"ok": False, "msg": "Qdrant недоступен — не переключаю"}
    settings.update({"VECTOR_BACKEND": target})
    vectorstore.milvus_reset_client()
    return {"ok": True, "restart": True,
            "msg": f"Активная векторная база переключена на «{target}». Перезапустите "
                   "сервис (кнопка «Перезапустить сервис»), чтобы поиск и индексация "
                   "начали использовать новый бэкенд."}


def redis_install() -> dict:
    """Установить и запустить Redis-сервер средствами ОС (apt/brew/apk/dnf/yum),
    включить REDIS_ENABLED и проверить подключение. Возвращает {ok, msg, log}."""
    import platform
    import shutil as _sh
    log: list[str] = []

    def run(cmd, **kw):
        log.append("$ " + " ".join(cmd))
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, **kw)
            out = (p.stdout or "") + (p.stderr or "")
            if out.strip():
                log.append(out.strip()[:4000])
            return p.returncode
        except Exception as e:
            log.append(f"[ошибка запуска] {e}")
            return 1

    # В Docker Redis — отдельный сервис compose (персистентный, restart:unless-stopped).
    # НЕ ставим Redis внутрь контейнера приложения: такой процесс гибнет при каждом
    # перезапуске контейнера — отсюда «после перезапуска недоступен, пока не нажмёшь
    # установить». Просто указываем сервис compose и проверяем доступность.
    if _in_docker():
        host = (settings.get("REDIS_HOST") or "").strip()
        if host.lower() in ("", "127.0.0.1", "localhost", "::1", "0.0.0.0"):
            host = "redis"                      # имя сервиса redis в сети compose
        port = int(settings.get("REDIS_PORT") or 6379)
        settings.update({"REDIS_ENABLED": True, "REDIS_HOST": host, "REDIS_PORT": port})
        if _redis_ping(host, port):
            return {"ok": True,
                    "msg": f"Redis (Docker): подключено к сервису «{host}:{port}»",
                    "log": f"Docker-режим: используется сервис compose «{host}» "
                           "(постоянный, авто-рестарт). Ping OK. REDIS_HOST зафиксирован."}
        return {"ok": False,
                "msg": f"Задан REDIS_HOST=«{host}», но сервис не отвечает. Поднимите Redis на "
                       "хосте: redis.cmd (или docker compose up -d — сервис redis есть в compose). "
                       "Внутрь контейнера приложения Redis не ставим — он не переживает перезапуск.",
                "log": f"Docker-режим: {host}:{port} не отвечает; ожидается сервис redis из compose."}

    # 0) уже доступен?
    if _redis_ping():
        settings.update({"REDIS_ENABLED": True})
        log.append("Redis уже запущен и отвечает на PING.")
        return {"ok": True, "msg": "Redis уже установлен и доступен", "log": "\n".join(log)}

    sysname = platform.system().lower()
    # 1) установка пакета Redis подходящим менеджером
    if _sh.which("brew"):                                   # macOS / Linuxbrew
        run(["brew", "install", "redis"])
    elif _sh.which("apt-get"):                              # Debian/Ubuntu
        run(["apt-get", "update"])
        run(["apt-get", "install", "-y", "redis-server"])
    elif _sh.which("apk"):                                  # Alpine (Docker)
        run(["apk", "add", "--no-cache", "redis"])
    elif _sh.which("dnf"):                                  # Fedora/RHEL
        run(["dnf", "install", "-y", "redis"])
    elif _sh.which("yum"):
        run(["yum", "install", "-y", "redis"])
    else:
        return {"ok": False,
                "msg": "не найден поддерживаемый менеджер пакетов (brew/apt/apk/dnf/yum)",
                "log": "\n".join(log) +
                "\nУстановите Redis вручную и включите REDIS_ENABLED в настройках."}

    # 2) запуск сервера (демонизированно; путь к бинарю ищем в PATH)
    redis_bin = _sh.which("redis-server")
    if _sh.which("brew") and "darwin" in sysname:
        run(["brew", "services", "start", "redis"])
    elif redis_bin:
        run([redis_bin, "--daemonize", "yes"])
    else:
        # системные службы Linux
        if _sh.which("systemctl"):
            run(["systemctl", "enable", "--now", "redis-server"]) or \
                run(["systemctl", "enable", "--now", "redis"])
        elif _sh.which("service"):
            run(["service", "redis-server", "start"])

    # 3) подождать и проверить
    for _ in range(10):
        if _redis_ping():
            break
        time.sleep(1)
    ok = _redis_ping()
    if ok:
        # включаем кэш и указываем локальный хост, если он ещё не задан явно
        changes = {"REDIS_ENABLED": True}
        if not (settings.get("REDIS_HOST") or "").strip():
            changes["REDIS_HOST"] = "127.0.0.1"
        settings.update(changes)
        log.append("Redis отвечает на PING. REDIS_ENABLED включён.")
        return {"ok": True, "msg": "Redis установлен, запущен и подключён",
                "log": "\n".join(log)}
    return {"ok": False,
            "msg": "Redis установлен, но не отвечает — запустите вручную (redis-server) "
                   "или проверьте права/службу",
            "log": "\n".join(log)}


def xtts_install() -> dict:
    """Установить пакет Coqui XTTS (клонирование голоса) в текущее окружение
    через pip. Тяжёлая зависимость (тянет torch). Возвращает {ok, msg, log}."""
    import sys
    log: list[str] = []

    def run(cmd, **kw):
        log.append("$ " + " ".join(cmd))
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, **kw)
            out = (p.stdout or "") + (p.stderr or "")
            if out.strip():
                log.append(out.strip()[-4000:])
            return p.returncode
        except Exception as e:
            log.append(f"[ошибка запуска] {e}")
            return 1

    # уже установлен?
    try:
        import importlib.util
        if importlib.util.find_spec("TTS") is not None:
            log.append("Пакет TTS уже установлен.")
            return {"ok": True, "msg": "Coqui XTTS уже установлен", "log": "\n".join(log)}
    except Exception:
        pass

    # ставим поддерживаемый форк coqui-tts в текущий интерпретатор (venv сервиса)
    rc = run([sys.executable, "-m", "pip", "install", "-U", "coqui-tts"])
    ok = False
    try:
        import importlib
        importlib.invalidate_caches()
        import importlib.util
        ok = importlib.util.find_spec("TTS") is not None
    except Exception:
        ok = (rc == 0)
    if ok:
        log.append("Coqui XTTS установлен. Модель скачается при первом синтезе.")
        return {"ok": True,
                "msg": "Coqui XTTS установлен. Загрузите образец голоса и выберите движок «xtts».",
                "log": "\n".join(log)}
    return {"ok": False,
            "msg": "не удалось установить coqui-tts — проверьте доступ к PyPI и версию Python "
                   "(нужен 3.9–3.12)",
            "log": "\n".join(log)}


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
    acc: dict = {}        # source -> [running_len, [parts]]
    next_off = None
    for _ in range(200000):  # страховка от бесконечного цикла
        try:
            pts, next_off = vectorstore.scroll(limit=512, offset=next_off,
                                               with_payload=True, with_vectors=False)
        except Exception:
            break
        for p in pts:
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
        if next_off is None or not pts:
            break
    return {s: "\n\n".join(v[1])[:cap_per_file] for s, v in acc.items()}


# Сборка графа базы знаний: фоновая задача с прогрессом + кэш результата.
_KB_JOB = {"running": False, "scrolled": 0, "total": 0, "stage": "",
           "ok": None, "error": None, "result": None, "ts": 0.0, "max_nodes": 400}
_KB_TTL = 300.0   # сек жизни кэша


def _kb_total_points() -> int:
    try:
        return int(vectorstore.collection_info().get("points_count", 0) or 0)
    except Exception:
        pass
    return 0


def _kb_build(max_nodes: int) -> None:
    """Один проход scroll по Qdrant с обновлением прогресса; кладёт результат в кэш."""
    j = _KB_JOB
    files: dict = {}
    total_points = 0
    next_off = None
    online = True
    j.update(running=True, scrolled=0, total=_kb_total_points(), stage="чтение индекса",
             ok=None, error=None)
    try:
        for _ in range(200000):
            try:
                pts, next_off = vectorstore.scroll(limit=512, offset=next_off,
                                                   with_payload=True, with_vectors=False)
            except Exception as e:
                online = False
                j["error"] = str(e)[:160]
                break
            for p in pts:
                pl = p.get("payload") or {}
                src = pl.get("source")
                if not src:
                    continue
                total_points += 1
                f = files.get(src)
                if f is None:
                    f = {"chunks": 0, "category": pl.get("doc_category") or "без категории",
                         "pages": set(), "is_org": False, "department": ""}
                    files[src] = f
                f["chunks"] += 1
                if pl.get("page") is not None:
                    f["pages"].add(pl.get("page"))
                if pl.get("doc_category") and f["category"] == "без категории":
                    f["category"] = pl.get("doc_category")
                if pl.get("org"):
                    f["is_org"] = True
                    if pl.get("department") and not f["department"]:
                        f["department"] = pl.get("department")
            j["scrolled"] = total_points
            if next_off is None or not pts:
                break

        j["stage"] = "построение графа"

        # Кластер: для карточек сотрудников — отдел, иначе категория документа.
        def _cluster(f):
            if f.get("is_org"):
                return f.get("department") or "Сотрудники (без отдела)"
            return f.get("category") or "без категории"

        cats: dict = {}
        for f in files.values():
            cats[_cluster(f)] = cats.get(_cluster(f), 0) + 1

        # Сотрудников показываем полностью (каждый — отдельная сущность), документы —
        # топ по числу фрагментов; в сумме не больше лимита.
        org_items = sorted([kv for kv in files.items() if kv[1].get("is_org")],
                           key=lambda kv: kv[0])
        doc_items = sorted([kv for kv in files.items() if not kv[1].get("is_org")],
                           key=lambda kv: kv[1]["chunks"], reverse=True)
        keep_org = org_items[:max_nodes]
        keep_doc = doc_items[:max(0, max_nodes - len(keep_org))]
        items = keep_org + keep_doc
        truncated = len(files) > len(items)

        nodes, links, used_cats = [], [], set()
        for src, f in items:
            clu = _cluster(f)
            label = src.split(" — ")[0] if f.get("is_org") else (os.path.basename(src) or src)
            nodes.append({"id": "f:" + src, "label": label,
                          "type": ("employee" if f.get("is_org") else "file"),
                          "category": clu, "chunks": f["chunks"],
                          "pages": len(f["pages"]), "source": src})
            links.append({"source": "c:" + clu, "target": "f:" + src})
            used_cats.add(clu)
        for c in used_cats:
            nodes.append({"id": "c:" + c, "label": c, "type": "category",
                          "files": cats.get(c, 0)})
        j["result"] = {"online": online, "nodes": nodes, "links": links,
                       "stats": {"files": len(files), "shown_files": len(items),
                                 "employees": len(org_items),
                                 "categories": len(cats), "chunks": total_points,
                                 "truncated": truncated, "max_nodes": max_nodes}}
        j["ts"] = time.time()
        j["ok"] = online
    except Exception as e:
        j["ok"] = False
        j["error"] = str(e)[:200]
    finally:
        j["running"] = False
        j["stage"] = ""


def kb_graph(max_nodes: int = 800, force: bool = False) -> dict:
    """Вернуть граф базы знаний. Если есть свежий кэш — отдаём сразу; иначе запускаем
    фоновую сборку и возвращаем {building:true} с прогрессом (клиент опрашивает
    kb_graph_status). force=True игнорирует кэш."""
    j = _KB_JOB
    now = time.time()
    if j["running"]:
        return {"building": True, "progress": _kb_progress()}
    fresh = (j["result"] and j["max_nodes"] == max_nodes and (now - j["ts"]) < _KB_TTL)
    if fresh and not force:
        return {"building": False, "cached": True,
                "age_sec": int(now - j["ts"]), **j["result"]}
    j["max_nodes"] = max_nodes
    threading.Thread(target=_kb_build, args=(max_nodes,), daemon=True).start()
    return {"building": True, "progress": {"scrolled": 0, "total": _KB_JOB.get("total", 0),
                                           "stage": "запуск"}}


def _kb_progress() -> dict:
    j = _KB_JOB
    tot = j.get("total", 0)
    pct = round(j["scrolled"] * 100.0 / tot, 1) if tot else None
    return {"scrolled": j.get("scrolled", 0), "total": tot, "pct": pct,
            "stage": j.get("stage", "")}


def kb_graph_status() -> dict:
    """Состояние сборки графа + результат, когда готов."""
    j = _KB_JOB
    if j["running"]:
        return {"building": True, "progress": _kb_progress()}
    if j["result"]:
        return {"building": False, "done": True, "ok": j["ok"], "error": j.get("error"),
                "cached": True, "age_sec": int(time.time() - j["ts"]), **j["result"]}
    return {"building": False, "done": False, "ok": j.get("ok"), "error": j.get("error")}


def kb_search(q: str, limit: int = 400) -> dict:
    """Поиск по словам в графе базы знаний: находит файлы (узлы), где встречаются
    слова запроса — по тексту чанков (семантически + буквально) и по именам файлов.
    Возвращает список источников с числом совпадений — фронт подсвечивает их в графе.
    Работает поверх активной векторной базы (Qdrant/Milvus) через vectorstore."""
    q = (q or "").strip()
    if not q:
        return {"ok": True, "query": "", "sources": [], "count": 0}
    words = [w for w in q.lower().split() if len(w) >= 2]
    agg: dict = {}

    def _bump(src, *, chunks=0, wh=0, score=0.0, name=False):
        a = agg.setdefault(src, {"source": src, "chunks": 0, "word_hits": 0,
                                 "score": 0.0, "name": False})
        a["chunks"] += chunks
        a["word_hits"] += wh
        a["score"] = max(a["score"], score)
        a["name"] = a["name"] or name

    # 1) семантический + буквальный поиск по тексту чанков
    try:
        import retriever
        qv = retriever._embed_query(q)
        hits = vectorstore.search(qv, int(limit), with_payload=True)
        for h in hits:
            pl = h.get("payload") or {}
            src = pl.get("source")
            if not src:
                continue
            txt = (pl.get("text") or "").lower()
            wm = sum(1 for w in words if w in txt)
            _bump(src, chunks=1, wh=wm, score=float(h.get("score") or 0))
    except Exception as e:
        # без эмбеддера семантику пропускаем — остаётся поиск по именам файлов
        pass

    # 2) совпадение слов в именах файлов (источников)
    try:
        for src in vectorstore.list_values("source", 100000):
            low = str(src).lower()
            if any(w in low for w in words):
                _bump(src, wh=1, name=True)
    except Exception:
        pass

    res = sorted(agg.values(),
                 key=lambda x: (x["word_hits"], x["chunks"], x["score"]), reverse=True)
    return {"ok": True, "query": q, "words": words,
            "sources": res[:200], "count": len(res),
            "backend": vectorstore.backend()}


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

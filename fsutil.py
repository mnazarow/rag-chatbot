"""Устойчивый обход файловой системы.

`Path.rglob` падает целиком, если хоть один каталог недоступен (например
OSError [Errno 5] Input/output error на сетевой/повреждённой папке, битый симлинк,
нет прав). Эти функции используют `os.walk(onerror=…)`: проблемный каталог
пропускается, обход продолжается — индексация не срывается из-за одной плохой папки.
"""
from __future__ import annotations
import os
from pathlib import Path


def walk_files(root, onerror=None, exclude_dirs=None):
    """Рекурсивно отдаёт все файлы внутри root как Path, устойчиво к ошибкам I/O.

    onerror(err) — необязательный колбэк для недоступных каталогов; если не задан,
    путь печатается в stdout. Обход в любом случае продолжается.
    exclude_dirs — множество имён каталогов, поддеревья которых пропускаются целиком
    (напр. {"telegram"} — файлы TG-обучения индексируются отдельно tg_train с payload
    tg=True/tg_chat_id, и общий обход не должен перезаписывать их без этих меток).
    """
    root = Path(root)
    excl = {d.lower() for d in (exclude_dirs or ())}

    def _err(err):
        if onerror is not None:
            try:
                onerror(err)
                return
            except Exception:
                pass
        path = getattr(err, "filename", "") or ""
        print(f"  ! пропущен недоступный путь: {path or err}")

    for dirpath, dirnames, filenames in os.walk(root, onerror=_err, followlinks=False):
        if excl:
            # правим dirnames на месте — os.walk не будет спускаться в исключённые
            dirnames[:] = [d for d in dirnames if d.lower() not in excl]
        for name in filenames:
            yield Path(dirpath) / name


def iter_doc_files(root, suffixes, onerror=None, exclude_dirs=None):
    """Файлы с подходящими расширениями (set/iterable, с точкой и в нижнем регистре),
    устойчиво к ошибкам ввода-вывода."""
    sfx = {s.lower() for s in suffixes}
    for p in walk_files(root, onerror=onerror, exclude_dirs=exclude_dirs):
        if p.suffix.lower() in sfx:
            yield p

"""Устойчивый обход файловой системы.

`Path.rglob` падает целиком, если хоть один каталог недоступен (например
OSError [Errno 5] Input/output error на сетевой/повреждённой папке, битый симлинк,
нет прав). Эти функции используют `os.walk(onerror=…)`: проблемный каталог
пропускается, обход продолжается — индексация не срывается из-за одной плохой папки.
"""
from __future__ import annotations
import os
from pathlib import Path


def walk_files(root, onerror=None):
    """Рекурсивно отдаёт все файлы внутри root как Path, устойчиво к ошибкам I/O.

    onerror(err) — необязательный колбэк для недоступных каталогов; если не задан,
    путь печатается в stdout. Обход в любом случае продолжается.
    """
    root = Path(root)

    def _err(err):
        if onerror is not None:
            try:
                onerror(err)
                return
            except Exception:
                pass
        path = getattr(err, "filename", "") or ""
        print(f"  ! пропущен недоступный путь: {path or err}")

    for dirpath, _dirnames, filenames in os.walk(root, onerror=_err, followlinks=False):
        for name in filenames:
            yield Path(dirpath) / name


def iter_doc_files(root, suffixes, onerror=None):
    """Файлы с подходящими расширениями (set/iterable, с точкой и в нижнем регистре),
    устойчиво к ошибкам ввода-вывода."""
    sfx = {s.lower() for s in suffixes}
    for p in walk_files(root, onerror=onerror):
        if p.suffix.lower() in sfx:
            yield p

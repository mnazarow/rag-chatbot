"""Загрузчики документов: каждый возвращает текст (+ при наличии — постранично).

Поддержка: PDF, DOCX, PPTX, XLSX/CSV (прайс-листы), TXT/MD, HTML,
аудио/видео (обучающие записи -> транскрибация Whisper).
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterator
import settings

# Какие расширения к какому обработчику
AUDIO_VIDEO = {".mp3", ".wav", ".m4a", ".aac", ".mp4", ".mov", ".mkv", ".webm"}


def load_file(path: Path) -> Iterator[dict]:
    """Yield {'text', 'page'} для одного файла. Пустые куски пропускаются."""
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            yield from _load_pdf(path)
        elif ext == ".docx":
            yield from _load_docx(path)
        elif ext == ".pptx":
            yield from _load_pptx(path)
        elif ext in {".xlsx", ".xls", ".csv"}:
            yield from _load_table(path)
        elif ext in {".txt", ".md"}:
            yield {"text": path.read_text(errors="ignore"), "page": None}
        elif ext in {".html", ".htm"}:
            yield from _load_html(path)
        elif ext in {".dxf", ".dwg"}:
            yield from _load_cad(path)
        elif ext in AUDIO_VIDEO:
            yield from _load_av(path)
        # остальное молча пропускаем
    except Exception as e:  # один битый файл не должен ронять индексацию
        print(f"  ! ошибка чтения {path.name}: {e}")


def _load_pdf(path: Path):
    import fitz  # pymupdf
    doc = fitz.open(path)
    for i, page in enumerate(doc, 1):
        txt = page.get_text("text")
        if txt.strip():
            yield {"text": txt, "page": i}


def _load_docx(path: Path):
    import docx
    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    # таблицы внутри документа
    for tbl in d.tables:
        for row in tbl.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    if parts:
        yield {"text": "\n".join(parts), "page": None}


def _load_pptx(path: Path):
    from pptx import Presentation
    prs = Presentation(str(path))
    for i, slide in enumerate(prs.slides, 1):
        chunks = []
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                chunks.append(shape.text_frame.text)
        # заметки докладчика часто = расшифровка обучения
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
            chunks.append("Заметки: " + slide.notes_slide.notes_text_frame.text)
        if chunks:
            yield {"text": "\n".join(chunks), "page": i}


def _load_table(path: Path):
    """Прайс-листы и таблицы: каждую строку превращаем в 'колонка: значение'."""
    import pandas as pd
    if path.suffix.lower() == ".csv":
        frames = {"csv": pd.read_csv(path, dtype=str, keep_default_na=False)}
    else:
        frames = pd.read_excel(path, sheet_name=None, dtype=str)
    for sheet, df in frames.items():
        df = df.fillna("")
        rows = []
        for _, row in df.iterrows():
            pairs = [f"{col}: {val}" for col, val in row.items() if str(val).strip()]
            if pairs:
                rows.append("; ".join(pairs))
        if rows:
            yield {"text": f"Лист «{sheet}»\n" + "\n".join(rows), "page": None}


def _load_html(path: Path):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(path.read_text(errors="ignore"), "html.parser")
    text = soup.get_text(separator="\n")
    if text.strip():
        yield {"text": text, "page": None}


def _dwg_to_dxf(path: Path) -> Path:
    """Конвертировать DWG -> DXF (dwg2dxf из libredwg-tools или ODA File Converter)."""
    import shutil
    import subprocess
    import tempfile
    out = Path(tempfile.gettempdir()) / (path.stem + "_conv.dxf")
    if shutil.which("dwg2dxf"):
        subprocess.run(["dwg2dxf", "-o", str(out), str(path)],
                       check=True, capture_output=True, timeout=180)
        if out.exists():
            return out
    oda = shutil.which("ODAFileConverter")
    if oda:
        ind, outd = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        shutil.copy(path, ind / path.name)
        subprocess.run([oda, str(ind), str(outd), "ACAD2018", "DXF", "0", "1"],
                       capture_output=True, timeout=300)
        cand = outd / (path.stem + ".dxf")
        if cand.exists():
            shutil.copy(cand, out)
            return out
    raise RuntimeError("для DWG нужен конвертер в DXF: установите libredwg-tools "
                       "(команда dwg2dxf) или ODA File Converter, либо сохраните "
                       "чертёж как DXF/PDF")


def _load_cad(path: Path):
    """Извлечь весь текст из чертежа DXF/DWG: TEXT, MTEXT, атрибуты блоков,
    размеры, имена слоёв (для DWG требуется конвертация в DXF)."""
    import ezdxf
    src, tmp = path, None
    if path.suffix.lower() == ".dwg":
        src = _dwg_to_dxf(path)
        tmp = src
    try:
        try:
            doc = ezdxf.readfile(str(src))
        except Exception:
            from ezdxf import recover
            doc, _ = recover.readfile(str(src))

        lines = []

        def grab(container):
            for e in container:
                try:
                    t = e.dxftype()
                    s = ""
                    if t in ("TEXT", "ATTRIB", "ATTDEF"):
                        s = e.dxf.get("text", "")
                    elif t == "MTEXT":
                        s = e.text
                    elif t == "DIMENSION":
                        s = e.dxf.get("text", "")
                        if s in ("", "<>"):
                            s = ""
                    if s and s.strip():
                        lines.append(s.strip())
                    if t == "INSERT":  # атрибуты вставленных блоков
                        for a in getattr(e, "attribs", []) or []:
                            v = a.dxf.get("text", "")
                            if v and v.strip():
                                lines.append(v.strip())
                except Exception:
                    continue

        for layout in doc.layouts:          # модель + листы
            grab(layout)
        for blk in doc.blocks:              # текст внутри определений блоков
            grab(blk)

        try:
            layers = [ly.dxf.name for ly in doc.layers]
        except Exception:
            layers = []

        body = "\n".join(dict.fromkeys(lines))  # дедуп с сохранением порядка
        if layers:
            body += "\nСлои: " + ", ".join(layers[:300])
        body = body.strip()
        if body:
            yield {"text": body, "page": None}
    finally:
        if tmp:
            try:
                tmp.unlink()
            except Exception:
                pass


_FASTER_WHISPER = None  # ленивый кеш модели faster-whisper


def _load_av(path: Path):
    """Транскрибация аудио/видео. Бэкенд зависит от настройки WHISPER_BACKEND:
       mlx    — Apple Metal (mlx-whisper),
       faster — GPU/CPU (faster-whisper, CTranslate2)."""
    print(f"  ~ транскрибирую {path.name} ...")
    device = settings.get("DEVICE")
    model = settings.get("WHISPER_MODEL")
    if settings.get("WHISPER_BACKEND") == "faster":
        global _FASTER_WHISPER
        if _FASTER_WHISPER is None:
            from faster_whisper import WhisperModel
            compute = "float16" if device == "cuda" else "int8"
            _FASTER_WHISPER = WhisperModel(model, device=device, compute_type=compute)
        segments, _ = _FASTER_WHISPER.transcribe(str(path))
        text = " ".join(s.text for s in segments)
        if text.strip():
            yield {"text": text, "page": None}
    else:  # mlx
        import mlx_whisper
        result = mlx_whisper.transcribe(str(path), path_or_hf_repo=model)
        if result.get("text", "").strip():
            yield {"text": result["text"], "page": None}

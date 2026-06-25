"""Загрузчики документов: каждый возвращает текст (+ при наличии — постранично).

Поддержка: PDF, DOCX, PPTX, XLSX/CSV (прайс-листы), TXT/MD, HTML,
аудио/видео (обучающие записи -> транскрибация Whisper).
"""
from __future__ import annotations
import warnings
from pathlib import Path
from typing import Iterator
import settings

# шумные предупреждения парсеров (openpyxl Data Validation и т.п.)
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
warnings.filterwarnings("ignore", message=".*Data Validation extension.*")

# Какие расширения к какому обработчику
AUDIO_VIDEO = {".mp3", ".wav", ".m4a", ".aac", ".mp4", ".mov", ".mkv", ".webm"}
# RAW-фото камер: конвертируются в изображение → OCR → текстовый PDF
RAW_PHOTO = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".rw2", ".orf", ".sr2"}
# Растровые изображения: распознавание текста (OCR)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".jfif"}
# Архивы: распаковываются, содержимое индексируется как обычные файлы
ARCHIVE_EXTS = {".zip", ".rar", ".7z", ".tar", ".gz", ".tgz", ".bz2"}
_ARCHIVE_MAX_DEPTH = 2               # защита от вложенных архивов / «архивных бомб»
_ARCHIVE_MAX_FILES = 5000            # лимит файлов внутри одного архива
_ARCHIVE_MAX_BYTES = 4 * 1024 ** 3   # лимит суммарного распакованного объёма (4 ГБ)


def probe_file(path_str: str, timeout: int = 0):
    """Проверить, извлекается ли текст из файла. Возвращает (status, issue):
    status ∈ {'ok','failed','timeout'}, issue — текст проблемы или None.
    Используется пулом процессов для ПАРАЛЛЕЛЬНОЙ проверки каталога."""
    import signal
    p = Path(path_str)
    use_alarm = bool(timeout) and hasattr(signal, "SIGALRM")

    def _to(_s, _f):
        raise TimeoutError()
    try:
        if use_alarm:
            signal.signal(signal.SIGALRM, _to)
            signal.alarm(int(timeout))
        got = False
        for part in load_file(p):
            if (part.get("text", "") or "").strip():
                got = True
                break  # текст найден — дальше можно не читать
        if use_alarm:
            signal.alarm(0)
        return ("ok", None) if got else ("failed", "текст не извлечён")
    except TimeoutError:
        if use_alarm:
            signal.alarm(0)
        return ("timeout", f"превышен лимит {timeout} c")
    except Exception as e:
        if use_alarm:
            signal.alarm(0)
        return ("failed", str(e)[:200])


def load_file(path: Path, _depth: int = 0) -> Iterator[dict]:
    """Yield {'text', 'page'} для одного файла. Пустые куски пропускаются.
    `_depth` — внутренний счётчик вложенности для распаковки архивов."""
    ext = path.suffix.lower()
    try:
        if ext in ARCHIVE_EXTS:
            yield from _load_archive(path, _depth)
            return
        if ext == ".pdf":
            yield from _load_pdf(path)
        elif ext == ".docx":
            yield from _load_docx(path)
        elif ext == ".pptx":
            yield from _load_pptx(path)
        elif ext in {".xlsx", ".xlsm", ".xls", ".csv"}:
            yield from _load_table(path)
        elif ext in {".txt", ".md"}:
            yield {"text": path.read_text(errors="ignore"), "page": None}
        elif ext in {".html", ".htm", ".mhtml", ".mht"}:
            yield from _load_html(path)
        elif ext == ".doc":
            yield from _load_doc(path)
        elif ext in {".xml"}:
            yield from _load_xml(path)
        elif ext == ".json":
            yield from _load_json(path)
        elif ext == ".url":
            yield from _load_url(path)
        elif ext == ".msg":
            yield from _load_msg(path)
        elif ext == ".svg":
            yield from _load_svg(path)
        elif ext in {".dxf", ".dwg", ".stp", ".step", ".igs", ".iges"}:
            # чертежи/3D-CAD — тяжёлая конвертация DWG; можно отключить ради скорости
            if _enabled("PARSE_CAD"):
                if ext in {".dxf", ".dwg"}:
                    yield from _load_cad(path)
                else:
                    yield from _load_cad_exchange(path)
        elif ext in IMAGE_EXTS:
            if _enabled("OCR_IMAGES"):   # OCR изображений — самый долгий этап
                yield from _load_image(path)
        elif ext in RAW_PHOTO:
            if _enabled("OCR_RAW"):
                yield from _load_raw(path)
        elif ext in AUDIO_VIDEO:
            if _enabled("TRANSCRIBE_AV"):  # транскрибация Whisper — минуты на файл
                yield from _load_av(path)
        # остальное молча пропускаем
    except Exception as e:  # один битый файл не должен ронять индексацию
        print(f"  ! ошибка чтения {path.name}: {e}")


def _enabled(key: str) -> bool:
    """Включён ли тяжёлый экстрактор (по рантайм-настройке, по умолчанию True)."""
    try:
        v = settings.get(key)
        return True if v is None else bool(v)
    except Exception:
        return True


def _ocr_pdf_page(page, i):
    """Распознать «картиночную» страницу PDF (текст нарисован графикой): рендерим
    страницу в изображение и прогоняем через OCR. Возвращает {'text','page'}."""
    try:
        import fitz  # pymupdf
        from PIL import Image
        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))  # ~180 DPI для читабельности
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        for part in _ocr_image(img):
            if part.get("text", "").strip():
                yield {"text": part["text"], "page": i}
    except Exception as e:
        print(f"  ~ OCR страницы PDF {i} не удался: {e}")


def _load_pdf(path: Path):
    import fitz  # pymupdf
    doc = fitz.open(path)
    # OCR страниц без текстового слоя (сканы/дизайн-страницы) — по флагу OCR_IMAGES
    ocr_on = _enabled("OCR_IMAGES")
    for i, page in enumerate(doc, 1):
        txt = page.get_text("text")
        if txt.strip():
            yield {"text": txt, "page": i}
        # мало или нет текста — вероятно, страница нарисована картинкой: распознаём
        if ocr_on and len(txt.strip()) < 25 and _ocr_available():
            yield from _ocr_pdf_page(page, i)


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
    ext = path.suffix.lower()
    if ext in {".mhtml", ".mht"}:
        # MHTML — это MIME-архив; вытаскиваем html-часть
        import email
        msg = email.message_from_bytes(path.read_bytes())
        html = ""
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                payload = part.get_payload(decode=True) or b""
                charset = part.get_content_charset() or "utf-8"
                html += payload.decode(charset, errors="ignore")
        raw = html or path.read_text(errors="ignore")
    else:
        raw = path.read_text(errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
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


def _load_cad_exchange(path: Path):
    """Текстовые метаданные из STEP (.stp/.step) и IGES (.igs/.iges):
    названия деталей/изделий, описания, заголовок, единицы, автор.
    Геометрия не извлекается (для текстового поиска она бесполезна)."""
    ext = path.suffix.lower()
    text = path.read_text(errors="ignore")
    body = _parse_iges(text) if ext in (".igs", ".iges") else _parse_step(text)
    if body.strip():
        yield {"text": body, "page": None}


def _parse_step(t: str) -> str:
    import re
    seen, names = set(), []
    # все строковые литералы STEP в одинарных кавычках ('' = апостроф)
    for m in re.finditer(r"'((?:[^']|'')*)'", t):
        s = m.group(1).replace("''", "'").strip()
        if len(s) >= 2 and not s.isdigit() and re.search(r"[A-Za-zА-Яа-я0-9]", s):
            if s not in seen:
                seen.add(s)
                names.append(s)
    if not names:
        return ""
    return "Метаданные STEP (названия, описания, единицы, заголовок):\n" + "\n".join(names[:2000])


def _parse_iges(t: str) -> str:
    import re
    start, glob = [], []
    for line in t.splitlines():
        if len(line) >= 73:
            sec = line[72]
            if sec == "S":  # Start section — свободный текст-описание
                s = line[:72].strip()
                if s:
                    start.append(s)
            elif sec == "G":  # Global section — параметры (Hollerith-строки)
                glob.append(line[:72])
    gtext = "".join(glob)
    holler, i = [], 0
    while i < len(gtext):  # Hollerith: NNNNH<строка ровно NNNN символов>
        m = re.match(r"(\d+)H", gtext[i:])
        if m:
            n = int(m.group(1))
            pos = i + m.end()
            val = gtext[pos:pos + n].strip()
            if val and re.search(r"[A-Za-zА-Яа-я0-9]", val):
                holler.append(val)
            i = pos + n
        else:
            i += 1
    out = []
    if start:
        out.append("Описание (IGES Start):\n" + "\n".join(start))
    if holler:
        out.append("Параметры (IGES Global):\n" + ", ".join(holler))
    return "\n".join(out)


_OCR_OK = None  # кеш проверки доступности OCR


def _ocr_available() -> bool:
    """Установлены ли pytesseract + сам tesseract. Предупреждаем один раз."""
    global _OCR_OK
    if _OCR_OK is None:
        import importlib.util
        import shutil
        has_lib = importlib.util.find_spec("pytesseract") is not None
        has_bin = shutil.which("tesseract") is not None
        _OCR_OK = has_lib and has_bin
        if not _OCR_OK:
            miss = []
            if not has_lib:
                miss.append("pytesseract (pip install pytesseract Pillow)")
            if not has_bin:
                miss.append("tesseract (системный пакет, напр. tesseract-ocr + -rus)")
            print(f"  ~ OCR недоступен — пропускаю распознавание картинок/RAW. "
                  f"Не хватает: {', '.join(miss)}. Либо отключите OCR в настройках.")
    return _OCR_OK


def _ocr_lang():
    import pytesseract
    try:
        langs = pytesseract.get_languages(config="")
        if "rus" in langs:
            return "rus+eng" if "eng" in langs else "rus"
    except Exception:
        pass
    return "eng"


def _ocr_image(img):
    """PIL.Image → OCR → текстовый (searchable) PDF → извлечённый текст.
    Общий помощник для растровых изображений и RAW-фото."""
    import io
    import pytesseract
    import fitz  # pymupdf

    # уменьшаем огромные снимки — ускоряет OCR без потери читаемости текста
    maxdim = 3500
    if max(img.size) > maxdim:
        k = maxdim / max(img.size)
        img = img.resize((int(img.size[0] * k), int(img.size[1] * k)))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    pdf_bytes = pytesseract.image_to_pdf_or_hocr(img, lang=_ocr_lang(), extension="pdf")
    doc = fitz.open(stream=io.BytesIO(pdf_bytes).getvalue(), filetype="pdf")
    for i, page in enumerate(doc, 1):
        txt = page.get_text("text")
        if txt.strip():
            yield {"text": txt, "page": i}


def _load_image(path: Path):
    """Растровое изображение (jpg/png/…) → OCR. Полезно для сканов и фото
    документов, скриншотов прайсов и т. п."""
    if not _ocr_available():
        return  # нет tesseract/pytesseract — не декодируем картинку зря
    from PIL import Image

    print(f"  ~ распознаю (OCR) {path.name} ...")
    with Image.open(path) as img:
        img.load()
        yield from _ocr_image(img)


def _load_raw(path: Path):
    """RAW-фото (CR2 и др.) → изображение → OCR → текст.
    Полезно для сфотографированных документов."""
    if not _ocr_available():
        return
    import rawpy
    from PIL import Image

    print(f"  ~ распознаю (OCR) {path.name} ...")
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess()
    img = Image.fromarray(rgb)
    yield from _ocr_image(img)


def _load_svg(path: Path):
    """SVG — извлекаем текст из элементов <text>/<tspan>."""
    import re
    data = path.read_text(errors="ignore")
    # вытаскиваем содержимое текстовых тегов
    parts = re.findall(r"<(?:text|tspan)\b[^>]*>(.*?)</(?:text|tspan)>", data,
                       flags=re.DOTALL | re.IGNORECASE)
    text = "\n".join(re.sub(r"<[^>]+>", " ", p) for p in parts).strip()
    if text:
        yield {"text": text, "page": None}


def _load_xml(path: Path):
    """XML — собираем весь видимый текст из узлов."""
    try:
        import xml.etree.ElementTree as ET
        root = ET.parse(str(path)).getroot()
        parts = [t.strip() for t in root.itertext() if t and t.strip()]
        text = "\n".join(parts)
    except Exception:
        import re
        raw = path.read_text(errors="ignore")
        text = re.sub(r"<[^>]+>", " ", raw)
    if text.strip():
        yield {"text": text, "page": None}


def _load_json(path: Path):
    """JSON — плоское текстовое представление пар ключ/значение."""
    import json
    data = json.loads(path.read_text(errors="ignore"))

    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from walk(v, f"{prefix}{k}: ")
        elif isinstance(obj, list):
            for it in obj:
                yield from walk(it, prefix)
        else:
            s = str(obj).strip()
            if s:
                yield f"{prefix}{s}"

    text = "\n".join(walk(data))
    if text.strip():
        yield {"text": text, "page": None}


def _load_url(path: Path):
    """Ярлык .url (Windows Internet Shortcut) — извлекаем адрес ссылки."""
    import re
    data = path.read_text(errors="ignore")
    m = re.search(r"URL\s*=\s*(\S+)", data, flags=re.IGNORECASE)
    url = m.group(1).strip() if m else ""
    text = f"Ссылка ({path.stem}): {url}".strip()
    if url:
        yield {"text": text, "page": None}


def _load_doc(path: Path):
    """Старый Word (.doc) — конвертация через antiword или LibreOffice."""
    import shutil
    import subprocess
    import tempfile

    # 1) antiword — быстрый и точный для .doc
    if shutil.which("antiword"):
        try:
            out = subprocess.run(["antiword", str(path)], capture_output=True,
                                 timeout=120)
            text = out.stdout.decode("utf-8", errors="ignore")
            if text.strip():
                yield {"text": text, "page": None}
                return
        except Exception:
            pass

    # 2) LibreOffice/soffice — конвертируем .doc → .docx, читаем как docx
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        with tempfile.TemporaryDirectory() as td:
            try:
                subprocess.run([soffice, "--headless", "--convert-to", "docx",
                                "--outdir", td, str(path)],
                               capture_output=True, timeout=240)
                conv = Path(td) / (path.stem + ".docx")
                if conv.exists():
                    yield from _load_docx(conv)
                    return
            except Exception:
                pass
    print(f"  ! пропуск {path.name}: нет antiword/LibreOffice для .doc")


def _load_msg(path: Path):
    """Письмо Outlook (.msg) — тема, отправитель, тело."""
    import extract_msg
    m = extract_msg.Message(str(path))
    try:
        head = []
        for label, val in (("Тема", m.subject), ("От", m.sender),
                           ("Кому", m.to), ("Дата", m.date)):
            if val:
                head.append(f"{label}: {val}")
        body = (m.body or "").strip()
        text = ("\n".join(head) + "\n\n" + body).strip()
    finally:
        try:
            m.close()
        except Exception:
            pass
    if text:
        yield {"text": text, "page": None}


def _extract_archive(path: Path, dest: Path) -> bool:
    """Распаковать архив в каталог dest. Сначала пробуем библиотеки Python,
    затем системные утилиты (7z/bsdtar/unar). Возвращает True при успехе."""
    import shutil
    import subprocess
    ext = path.suffix.lower()

    # 1) Python-библиотеки
    try:
        if ext == ".zip":
            import zipfile
            with zipfile.ZipFile(path) as z:
                total = sum(i.file_size for i in z.infolist())
                if total > _ARCHIVE_MAX_BYTES:
                    raise RuntimeError("распакованный объём превышает лимит")
                z.extractall(dest)
            return True
        if ext == ".7z":
            import py7zr
            with py7zr.SevenZipFile(path, "r") as z:
                z.extractall(dest)
            return True
        if ext == ".rar":
            import rarfile
            with rarfile.RarFile(path) as r:
                r.extractall(dest)
            return True
        if ext in {".tar", ".tgz"} or (ext in {".gz", ".bz2"} and ".tar" in path.name.lower()):
            import tarfile
            with tarfile.open(path) as t:
                try:
                    t.extractall(dest, filter="data")  # защита от path-traversal (3.12+)
                except TypeError:
                    t.extractall(dest)  # старые версии Python без параметра filter
            return True
        if ext in {".gz", ".bz2"}:
            # одиночный поток (не tar-архив) — распаковываем в один файл
            import bz2
            import gzip
            opener = gzip.open if ext == ".gz" else bz2.open
            out = Path(dest) / path.stem  # отбрасываем .gz/.bz2
            with opener(path, "rb") as src, open(out, "wb") as dst:
                while True:
                    block = src.read(1 << 20)
                    if not block:
                        break
                    dst.write(block)
            return True
    except ImportError:
        pass  # нужной библиотеки нет — пробуем системные утилиты
    except Exception as e:
        print(f"  ! {path.name}: ошибка распаковки ({e}); пробую системные утилиты")

    # 2) системные утилиты (best-effort)
    for tool in (["7z", "x", "-y", f"-o{dest}", str(path)],
                 ["7za", "x", "-y", f"-o{dest}", str(path)],
                 ["bsdtar", "-xf", str(path), "-C", str(dest)],
                 ["unar", "-quiet", "-output-directory", str(dest), str(path)]):
        if shutil.which(tool[0]):
            try:
                r = subprocess.run(tool, capture_output=True, timeout=900)
                if r.returncode == 0:
                    return True
            except Exception:
                continue
    return False


def _load_archive(path: Path, depth: int):
    """Распаковать архив и проиндексировать содержимое как обычные файлы.
    Внутренние файлы помечаются их путём внутри архива для цитирования."""
    if depth >= _ARCHIVE_MAX_DEPTH:
        print(f"  ! пропуск вложенного архива {path.name}: слишком глубоко")
        return
    import tempfile
    print(f"  ~ распаковываю архив {path.name} ...")
    with tempfile.TemporaryDirectory() as td:
        dest = Path(td)
        if not _extract_archive(path, dest):
            print(f"  ! не удалось распаковать {path.name}: "
                  f"нет py7zr/rarfile или утилит 7z/bsdtar/unar")
            return
        n = 0
        for inner in sorted(dest.rglob("*")):
            if not inner.is_file():
                continue
            n += 1
            if n > _ARCHIVE_MAX_FILES:
                print(f"  ! {path.name}: слишком много файлов в архиве, остановка")
                break
            rel = inner.relative_to(dest)
            try:
                for part in load_file(inner, _depth=depth + 1):
                    txt = part.get("text", "")
                    if txt.strip():
                        # помечаем источник внутри архива — пригодится для цитат
                        yield {"text": f"[{path.name} → {rel}]\n{txt}",
                               "page": part.get("page")}
            except Exception as e:
                print(f"  ! {path.name} → {rel}: {e}")


_FASTER_WHISPER = None  # ленивый кеш модели faster-whisper


def _av_windows(segments):
    """Сгруппировать сегменты Whisper (start/end/text) в окна ~CHUNK_SIZE символов,
    сохраняя тайминги начала/конца окна — чтобы потом показать кадр/фрагмент видео."""
    try:
        win = int(settings.get("CHUNK_SIZE"))
    except Exception:
        win = 900
    buf, t0, t1, n = [], None, None, 0
    for seg in segments:
        st = float(seg.get("start") or 0.0)
        en = float(seg.get("end") or st)
        tx = (seg.get("text") or "").strip()
        if not tx:
            continue
        if t0 is None:
            t0 = st
        t1 = en
        buf.append(tx)
        n += len(tx) + 1
        if n >= win:
            yield {"text": " ".join(buf), "page": None, "t_start": t0, "t_end": t1}
            buf, t0, t1, n = [], None, None, 0
    if buf:
        yield {"text": " ".join(buf), "page": None, "t_start": t0, "t_end": t1}


def _load_av(path: Path):
    """Транскрибация аудио/видео. Бэкенд зависит от настройки WHISPER_BACKEND:
       mlx    — Apple Metal (mlx-whisper),
       faster — GPU/CPU (faster-whisper, CTranslate2).
    Текст режется на окна с таймингами (t_start/t_end) для выдачи кадров/фрагментов."""
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
        segs = ({"start": s.start, "end": s.end, "text": s.text} for s in segments)
        yield from _av_windows(segs)
    else:  # mlx
        import mlx_whisper
        result = mlx_whisper.transcribe(str(path), path_or_hf_repo=model)
        segs = result.get("segments") or []
        if segs:
            yield from _av_windows(segs)
        elif result.get("text", "").strip():
            yield {"text": result["text"], "page": None}

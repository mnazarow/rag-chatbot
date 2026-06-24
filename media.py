"""Выдача исходных артефактов в ответах: изображения, чертежи CAD, кадры и
фрагменты видео. Превью генерируются «лениво» при первом запросе и кешируются
в папке previews/ — переиндексация для этого не нужна.

Тяжёлые зависимости (Pillow, rawpy, ezdxf, matplotlib, ffmpeg) подключаются по
месту; при их отсутствии превью просто не создаётся (отдаётся оригинал/ничего).
"""
from __future__ import annotations
import hashlib
import shutil
import subprocess
from pathlib import Path

import settings

ROOT = Path(__file__).resolve().parent
PREVIEW_DIR = ROOT / "previews"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".jfif"}
RAW_EXTS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".rw2", ".orf", ".sr2"}
CAD_EXTS = {".dxf", ".dwg"}
MODEL_EXTS = {".stp", ".step", ".igs", ".iges"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac"}

_CLIP_MAX_SEC = 600  # ограничение на длину вырезаемого фрагмента


def kind_of(source: str) -> str:
    """Категория артефакта: image|raw|cad|model|video|audio|pdf|other."""
    ext = Path(source).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in RAW_EXTS:
        return "raw"
    if ext in CAD_EXTS:
        return "cad"
    if ext in MODEL_EXTS:
        return "model"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext == ".pdf":
        return "pdf"
    return "other"


def has_preview(source: str) -> bool:
    """Можно ли построить визуальное превью для этого типа."""
    return kind_of(source) in ("image", "raw", "cad", "video", "pdf")


def resolve(source: str) -> Path | None:
    """Безопасно превратить относительный source в абсолютный путь внутри DOCS_DIR."""
    docs = Path(settings.get("DOCS_DIR")).expanduser().resolve()
    clean = [s for s in str(source or "").replace("\\", "/").split("/")
             if s not in ("", ".", "..")]
    if not clean:
        return None
    try:
        p = (docs / Path(*clean)).resolve()
    except Exception:
        return None
    if (p == docs or docs in p.parents) and p.is_file():
        return p
    return None


def _cache_path(source: str, tag: str, ext: str) -> Path:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(f"{source}|{tag}".encode("utf-8")).hexdigest()[:20]
    return PREVIEW_DIR / f"{h}{ext}"


def _save_image_thumb(img, out: Path, maxdim: int = 900) -> Path | None:
    try:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.thumbnail((maxdim, maxdim))
        img.save(out, "JPEG", quality=85)
        return out
    except Exception:
        return None


def _thumb_image(src: Path, out: Path) -> Path | None:
    try:
        from PIL import Image
        with Image.open(src) as img:
            img.load()
            return _save_image_thumb(img, out)
    except Exception:
        return None


def _thumb_raw(src: Path, out: Path) -> Path | None:
    try:
        import rawpy
        from PIL import Image
        with rawpy.imread(str(src)) as raw:
            rgb = raw.postprocess()
        return _save_image_thumb(Image.fromarray(rgb), out)
    except Exception:
        return None


def _thumb_cad(src: Path, out: Path) -> Path | None:
    """Рендер DXF/DWG в PNG через ezdxf + matplotlib (Agg, без дисплея)."""
    try:
        import ezdxf
        from ezdxf.addons.drawing import RenderContext, Frontend
        from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    dxf = src
    tmp_dxf = None
    if src.suffix.lower() == ".dwg":
        try:
            import loaders
            dxf = loaders._dwg_to_dxf(src)  # dwg2dxf/ODA, если установлены
            tmp_dxf = dxf
        except Exception:
            return None
    try:
        doc = ezdxf.readfile(str(dxf))
        msp = doc.modelspace()
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        Frontend(RenderContext(doc), MatplotlibBackend(ax)).draw_layout(msp, finalize=True)
        fig.savefig(str(out), dpi=120, facecolor="white")
        plt.close(fig)
        return out if out.exists() else None
    except Exception:
        return None
    finally:
        if tmp_dxf is not None:
            try:
                Path(tmp_dxf).unlink(missing_ok=True)
            except Exception:
                pass


def _thumb_pdf(src: Path, out: Path, page: int = 0) -> Path | None:
    """Рендер страницы PDF в PNG через PyMuPDF."""
    try:
        import fitz  # pymupdf
        doc = fitz.open(str(src))
        if doc.page_count == 0:
            return None
        pg = doc[max(0, min(page, doc.page_count - 1))]
        pix = pg.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        pix.save(str(out))
        return out if out.exists() else None
    except Exception:
        return None


def _thumb_video_frame(src: Path, out: Path, t: float = 1.0) -> Path | None:
    if not shutil.which("ffmpeg"):
        return None
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-ss", str(max(0.0, t)), "-i", str(src),
             "-frames:v", "1", "-vf", "scale=720:-1", str(out)],
            capture_output=True, timeout=120)
        return out if (r.returncode == 0 and out.exists()) else None
    except Exception:
        return None


def thumbnail(source: str, t: float | None = None) -> Path | None:
    """Путь к превью (создаёт и кеширует при первом обращении). None — нельзя."""
    src = resolve(source)
    if src is None:
        return None
    k = kind_of(source)
    if k == "video":
        tt = 1.0 if t is None else max(0.0, float(t))
        out = _cache_path(source, f"frame-{tt:.1f}", ".jpg")
        if out.exists():
            return out
        return _thumb_video_frame(src, out, tt)
    if k == "pdf":
        page = int(t) if t else 0
        out = _cache_path(source, f"pdf-p{page}", ".png")
        if out.exists():
            return out
        return _thumb_pdf(src, out, page)
    out = _cache_path(source, "thumb", ".jpg" if k in ("image", "raw") else ".png")
    if out.exists():
        return out
    if k == "image":
        return _thumb_image(src, out)
    if k == "raw":
        return _thumb_raw(src, out)
    if k == "cad":
        return _thumb_cad(src, out.with_suffix(".png"))
    return None


def clip(source: str, start: float, end: float) -> Path | None:
    """Вырезать фрагмент видео [start, end] (перекодировка для точности)."""
    if kind_of(source) != "video" or not shutil.which("ffmpeg"):
        return None
    src = resolve(source)
    if src is None:
        return None
    try:
        start = max(0.0, float(start))
        end = float(end)
    except Exception:
        return None
    dur = end - start
    if dur <= 0 or dur > _CLIP_MAX_SEC:
        return None
    out = _cache_path(source, f"clip-{start:.1f}-{end:.1f}", ".mp4")
    if out.exists():
        return out
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start), "-i", str(src), "-t", str(dur),
             "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac",
             "-movflags", "+faststart", str(out)],
            capture_output=True, timeout=600)
        return out if (r.returncode == 0 and out.exists()) else None
    except Exception:
        return None


def cite(source: str, page=None, t_start=None, t_end=None,
         score=None, category=None) -> dict:
    """Обогатить ссылку-источник типом артефакта и URL для выдачи в чате."""
    k = kind_of(source)
    exists = resolve(source) is not None
    item = {"source": source, "page": page, "kind": k, "exists": exists,
            "score": score, "category": category}
    if not exists:
        return item
    from urllib.parse import quote
    qs = quote(source)
    item["file_url"] = f"/api/media/file?source={qs}"
    if k in ("image", "raw", "cad", "pdf"):
        item["thumb_url"] = f"/api/media/thumb?source={qs}"
    if k == "video":
        # кадр по таймстампу сегмента (если есть), иначе с 1-й секунды
        ts = t_start if isinstance(t_start, (int, float)) else 1.0
        item["thumb_url"] = f"/api/media/thumb?source={qs}&t={ts}"
        if isinstance(t_start, (int, float)):
            item["t_start"] = round(float(t_start), 1)
        if isinstance(t_end, (int, float)):
            item["t_end"] = round(float(t_end), 1)
        if isinstance(t_start, (int, float)) and isinstance(t_end, (int, float)):
            item["clip_url"] = f"/api/media/clip?source={qs}&start={t_start}&end={t_end}"
    if k == "audio":
        if isinstance(t_start, (int, float)):
            item["t_start"] = round(float(t_start), 1)
        if isinstance(t_end, (int, float)):
            item["t_end"] = round(float(t_end), 1)
    return item

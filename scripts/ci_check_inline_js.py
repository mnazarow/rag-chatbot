#!/usr/bin/env python3
"""Извлечь инлайн-JS из static/index.html во временный файл для `node --check`.

Печатает путь к собранному JS в stdout (последняя строка) — CI прогоняет через node.
"""
import pathlib
import re
import sys

html = pathlib.Path("static/index.html").read_text(encoding="utf-8")
# Закрывающий тег матчим с учётом регистра и возможных пробелов (</script >, </SCRIPT>).
# FIXME(review): по HTML-спеке <script> — raw-text элемент и завершается ПЕРВЫМ вхождением
# "</script"; если в самом JS встречается литерал "</script>" (нужно писать "<\/script>"),
# извлечение оборвётся преждевременно. Полностью корректно это решается только HTML-парсером.
scripts = re.findall(r"<script(?![^>]*\ssrc=)[^>]*>(.*?)</script\s*>", html, re.S | re.I)
if not scripts:
    print("no inline scripts found", file=sys.stderr)
    sys.exit(1)
out = pathlib.Path("build_inline.js")
out.write_text("\n;\n".join(scripts), encoding="utf-8")
print(str(out))

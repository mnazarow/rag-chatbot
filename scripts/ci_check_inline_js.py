#!/usr/bin/env python3
"""Извлечь инлайн-JS из static/index.html во временный файл для `node --check`.

Печатает путь к собранному JS в stdout (последняя строка) — CI прогоняет через node.
"""
import pathlib
import re
import sys

html = pathlib.Path("static/index.html").read_text(encoding="utf-8")
scripts = re.findall(r"<script(?![^>]*src=)[^>]*>(.*?)</script>", html, re.S)
if not scripts:
    print("no inline scripts found", file=sys.stderr)
    sys.exit(1)
out = pathlib.Path("build_inline.js")
out.write_text("\n;\n".join(scripts), encoding="utf-8")
print(str(out))

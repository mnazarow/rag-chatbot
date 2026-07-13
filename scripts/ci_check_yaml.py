#!/usr/bin/env python3
"""Проверить, что compose-файлы разбираются как валидный YAML."""
import sys

import yaml

FILES = [
    "windows_variant/docker/docker-compose.windows.yml",
    "windows_variant/docker/docker-compose.gpu.yml",
    "docker_variant/docker-compose.yml",
]

ok = True
for f in FILES:
    try:
        yaml.safe_load(open(f, encoding="utf-8"))
        print("OK", f)
    except Exception as e:  # noqa: BLE001
        ok = False
        print("FAIL", f, e, file=sys.stderr)
sys.exit(0 if ok else 1)

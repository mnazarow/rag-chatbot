"""Простой in-memory лимитер запросов на ключ (IP или session_id).

Алгоритм — токен-бакет: у каждого ключа бак ёмкостью `burst`, пополняемый со
скоростью `rps` токенов в секунду. Один запрос тратит один токен; если токенов
нет — запрос отклоняется, а вызывающему сообщается, через сколько секунд
повторить (Retry-After). Потокобезопасно (единый Lock, операции дешёвые).

Настройки читаются вызывающим из settings и передаются в configure():
  RATE_LIMIT_RPS   — устойчивая скорость (запросов/с). None/0 → лимитер выключен,
                     поведение приложения не меняется (без ограничений).
  RATE_LIMIT_BURST — ёмкость бака (разрешённый всплеск). По умолчанию = max(rps, 1).

FIXME(review): лимит действует ПО  ПРОЦЕССУ (per-process). При нескольких воркерах
uvicorn/gunicorn каждый воркер держит собственные баки, поэтому фактический общий
предел ≈ N_workers × RATE_LIMIT_RPS. Для строгого глобального лимита нужен общий
бэкенд (Redis/procshare, как у llm_queue).
"""
from __future__ import annotations

import threading
import time

# Максимум ключей в памяти — страховка от разрастания при потоке с многих IP.
_MAX_KEYS = 100_000
# Порог простоя для сборки мусора (сек): полностью пополненный давно неактивный бак
# можно удалить без потери информации.
_IDLE_GC_SEC = 3600.0


class RateLimiter:
    """Токен-бакет на ключ. Конфигурация меняется на лету через configure()."""

    __slots__ = ("_rps", "_burst", "_buckets", "_lock", "_last_gc")

    def __init__(self, rps: float | None = None, burst: float | None = None):
        self._lock = threading.Lock()
        self._buckets: dict[str, list] = {}   # key -> [tokens, last_monotonic]
        self._last_gc = time.monotonic()
        self._rps = 0.0
        self._burst = 1.0
        self.configure(rps, burst)

    # ------------------------------------------------------------------ config
    def configure(self, rps: float | None, burst: float | None) -> None:
        """Обновить скорость/ёмкость (например из настроек админки)."""
        try:
            new_rps = float(rps) if rps else 0.0
        except (TypeError, ValueError):
            new_rps = 0.0
        if new_rps < 0:
            new_rps = 0.0
        try:
            new_burst = float(burst) if burst else 0.0
        except (TypeError, ValueError):
            new_burst = 0.0
        if new_burst <= 0:
            new_burst = max(new_rps, 1.0)
        with self._lock:
            self._rps = new_rps
            self._burst = new_burst

    @property
    def enabled(self) -> bool:
        return self._rps > 0.0

    # ------------------------------------------------------------------- check
    def check(self, key: str) -> tuple[bool, float]:
        """Списать один токен для ключа.

        Возвращает (allowed, retry_after_sec). Если лимитер выключен — всегда
        (True, 0.0). retry_after_sec — сколько ждать до появления токена.
        """
        if self._rps <= 0.0:
            return True, 0.0
        now = time.monotonic()
        with self._lock:
            rps = self._rps
            burst = self._burst
            slot = self._buckets.get(key)
            if slot is None:
                # новый ключ — полный бак минус текущий запрос
                if len(self._buckets) >= _MAX_KEYS:
                    self._gc(now, force=True)
                self._buckets[key] = [burst - 1.0, now]
                return True, 0.0
            tokens, last = slot
            tokens = min(burst, tokens + (now - last) * rps)
            if tokens >= 1.0:
                slot[0] = tokens - 1.0
                slot[1] = now
                allowed, retry = True, 0.0
            else:
                slot[0] = tokens
                slot[1] = now
                need = 1.0 - tokens
                retry = need / rps if rps > 0 else 1.0
                allowed = False
            self._maybe_gc(now)
            return allowed, retry

    # ---------------------------------------------------------------------- gc
    def _maybe_gc(self, now: float) -> None:
        if now - self._last_gc >= 60.0:
            self._gc(now)

    def _gc(self, now: float, force: bool = False) -> None:
        """Удалить давно неактивные полностью пополненные баки. Под _lock."""
        self._last_gc = now
        burst = self._burst
        rps = self._rps
        dead = []
        for k, (tokens, last) in self._buckets.items():
            idle = now - last
            refilled = min(burst, tokens + idle * rps)
            if refilled >= burst and idle >= _IDLE_GC_SEC:
                dead.append(k)
        for k in dead:
            self._buckets.pop(k, None)
        # аварийная очистка при переполнении: удаляем самые старые по last
        if force and len(self._buckets) >= _MAX_KEYS:
            oldest = sorted(self._buckets.items(), key=lambda kv: kv[1][1])
            for k, _ in oldest[: max(1, len(oldest) // 10)]:
                self._buckets.pop(k, None)

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()

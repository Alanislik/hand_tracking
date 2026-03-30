# -*- coding: utf-8 -*-
"""
core/smoothing.py — EMA cursor smoother.

Pipeline per frame:
  raw pixel (x, y)
    → velocity cap   — hard-clamp sudden jumps (hand detected far from last pos)
    → EMA buffer     — exponential weighted average
    → deadzone       — suppress sub-pixel jitter
    → output (int x, int y)

Acceleration is intentionally REMOVED.  It was the root cause of
"cursor runs to screen edge" — dist**1.15 amplifies even tiny deltas.
A linear pipeline with a good velocity_cap is stable and responsive.
"""
from __future__ import annotations
import math
import threading
from typing import Tuple

import numpy as np

from config import SmoothingConfig


class CursorSmoother:
    def __init__(self, cfg: SmoothingConfig, screen_w: int, screen_h: int) -> None:
        self._cfg   = cfg
        self._sw    = screen_w
        self._sh    = screen_h
        self._lock  = threading.Lock()
        self._size  = max(1, cfg.ema_window)
        self._buf   = np.zeros((self._size, 2), dtype=np.float64)
        self._idx   = 0
        self._count = 0
        # Initialise to screen centre — prevents first-frame jump from (0,0)
        self._last_x = float(screen_w) / 2.0
        self._last_y = float(screen_h) / 2.0

    # ── public ────────────────────────────────────────────────

    def push(self, x: float, y: float) -> Tuple[int, int]:
        with self._lock:
            self._lazy_resize()

            # 1. Velocity cap — discard teleport jumps
            x, y = self._cap(x, y)

            # 2. EMA
            self._buf[self._idx] = (x, y)
            self._idx   = (self._idx + 1) % self._size
            self._count = min(self._count + 1, self._size)
            sx, sy = self._ema()

            # 3. Deadzone — ignore micro-jitter
            dx = sx - self._last_x
            dy = sy - self._last_y
            dz = float(self._cfg.deadzone_px)
            if abs(dx) < dz and abs(dy) < dz:
                return int(round(self._last_x)), int(round(self._last_y))

            # 4. Clamp to screen
            sx = max(0.0, min(float(self._sw - 1), sx))
            sy = max(0.0, min(float(self._sh - 1), sy))
            self._last_x = sx
            self._last_y = sy
            return int(round(sx)), int(round(sy))

    def resize(self, window: int) -> None:
        with self._lock:
            self._size = max(1, window)

    def clear(self) -> None:
        with self._lock:
            self._count  = 0
            self._idx    = 0
            self._buf[:] = 0.0
            self._last_x = float(self._sw) / 2.0
            self._last_y = float(self._sh) / 2.0

    # ── private ───────────────────────────────────────────────

    def _lazy_resize(self) -> None:
        if self._buf.shape[0] != self._size:
            self._buf   = np.zeros((self._size, 2), dtype=np.float64)
            self._idx   = 0
            self._count = 0

    def _cap(self, x: float, y: float) -> Tuple[float, float]:
        cap = float(self._cfg.velocity_cap)
        dx  = x - self._last_x
        dy  = y - self._last_y
        d   = math.hypot(dx, dy)
        if d > cap:
            s = cap / d
            x = self._last_x + dx * s
            y = self._last_y + dy * s
        return x, y

    def _ema(self) -> Tuple[float, float]:
        n = self._count
        if n == self._size:
            ordered = np.roll(self._buf, -self._idx, axis=0)
        else:
            ordered = self._buf[:n]
        if n == 1:
            return float(ordered[0, 0]), float(ordered[0, 1])
        w = np.exp(np.linspace(-2.0, 0.0, n))
        w /= w.sum()
        r  = w @ ordered
        return float(r[0]), float(r[1])

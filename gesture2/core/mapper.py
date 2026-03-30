# -*- coding: utf-8 -*-
"""
core/mapper.py
==============
Maps normalised landmark coords → screen pixel coords.

The core formula is a simple linear remap:

  rx = (nx - zone_x0) / zone_width          # 0..1 within active zone
  screen_x = clamp(rx * sensitivity, 0, 1) * screen_width

sensitivity = 1.0  →  active zone maps 1:1 to full screen
sensitivity = 1.5  →  cursor moves 1.5× faster than hand
sensitivity < 1.0  →  slower / more precise

Active zone defaults are tightened to [0.2, 0.3, 0.8, 0.8] so the
mapped region matches where a hand actually sits in front of a typical
webcam (centre-left to centre-bottom of frame).  User can override
via calibration (C key) or config file.
"""
from __future__ import annotations
from typing import Tuple

from config import CalibConfig


class CoordinateMapper:
    def __init__(self, calib: CalibConfig, screen_w: int, screen_h: int,
                 sensitivity: float = 1.0) -> None:
        self._calib = calib
        self._sw    = screen_w
        self._sh    = screen_h
        self._sens  = max(0.1, float(sensitivity))

    @property
    def sensitivity(self) -> float:
        return self._sens

    @sensitivity.setter
    def sensitivity(self, v: float) -> None:
        self._sens = max(0.1, min(5.0, float(v)))

    def map(self, nx: float, ny: float) -> Tuple[float, float]:
        z = self._calib.zone          # [x0, y0, x1, y1]  normalised 0-1

        zone_w = max(z[2] - z[0], 1e-6)
        zone_h = max(z[3] - z[1], 1e-6)

        # Remap hand position within zone → [0, 1]
        rx = (nx - z[0]) / zone_w
        ry = (ny - z[1]) / zone_h

        # Apply sensitivity and clamp
        rx = max(0.0, min(1.0, rx * self._sens))
        ry = max(0.0, min(1.0, ry * self._sens))

        return rx * self._sw, ry * self._sh

    def update_calib(self, zone: list) -> None:
        self._calib.zone = list(zone)

    def update_screen(self, w: int, h: int) -> None:
        self._sw = w
        self._sh = h

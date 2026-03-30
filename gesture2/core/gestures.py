# -*- coding: utf-8 -*-
"""
core/gestures.py
================
Pure gesture classification.
Input  : List[Landmark]  (21 MediaPipe hand landmarks, normalised)
Output : GestureLabel    (plain enum string)

Zero side-effects.  Zero timers.  Zero I/O.
The state machine (state_machine.py) handles debounce / transitions.
"""
from __future__ import annotations
import math
from enum import Enum
from typing import List

from core.tracker import Landmark


class GestureLabel(str, Enum):
    """Every possible gesture the classifier can emit."""
    NONE      = "none"
    MOVE      = "move"
    PINCH_IDX = "pinch_idx"    # thumb + index  → left-click candidate
    PINCH_MID = "pinch_mid"    # thumb + middle → right-click candidate
    SCROLL    = "scroll"
    FIST      = "fist"
    OPEN_PALM = "open_palm"    # keyboard toggle
    IDLE      = "idle"


def classify_hand(lm: List[Landmark], side: str,
                  pinch_idx_thr: float = 0.048,
                  pinch_mid_thr: float = 0.060) -> GestureLabel:
    """
    Classify a single hand into one GestureLabel.

    Parameters
    ----------
    lm            : 21 normalised landmarks
    side          : "Left" or "Right" (post-mirror)
    pinch_idx_thr : max normalised distance for index pinch
    pinch_mid_thr : max normalised distance for middle pinch
    """
    # ── finger extension ─────────────────────────────────────
    thumb_up = lm[4].x < lm[3].x if side == "Right" else lm[4].x > lm[3].x
    idx  = _tip_above(lm, 8,  6)
    mid  = _tip_above(lm, 12, 10)
    ring = _tip_above(lm, 16, 14)
    pink = _tip_above(lm, 20, 18)
    n_ext = sum([idx, mid, ring, pink])

    # ── pinch distances ───────────────────────────────────────
    d_idx = _dist(lm, 4, 8)
    d_mid = _dist(lm, 4, 12)

    pinch_idx = d_idx < pinch_idx_thr
    pinch_mid = d_mid < pinch_mid_thr

    # ── classification (priority order) ──────────────────────

    # 1. Open palm → keyboard
    if n_ext == 4 and thumb_up:
        return GestureLabel.OPEN_PALM

    # 2. Scroll → index + middle up, rest down, no pinch
    if idx and mid and not ring and not pink and not pinch_idx:
        return GestureLabel.SCROLL

    # 3. Pinch index → left-click
    if pinch_idx and idx and not mid:
        return GestureLabel.PINCH_IDX

    # 4. Pinch middle → right-click
    if pinch_mid and not idx:
        return GestureLabel.PINCH_MID

    # 5. Move → only index extended
    if idx and not mid:
        return GestureLabel.MOVE

    # 6. Fist → all curled
    if n_ext == 0:
        return GestureLabel.FIST

    return GestureLabel.IDLE


# ── blink classification ──────────────────────────────────────────────────────

class BlinkLabel(str, Enum):
    NONE  = "none"
    LEFT  = "left"
    RIGHT = "right"


def classify_blink(blendshapes,
                   threshold:   float = 0.40,
                   both_margin: float = 0.12) -> BlinkLabel:
    """
    Classify blendshapes into a blink label.
    Returns NONE when both eyes blink simultaneously (natural blink).
    """
    sl = sr = 0.0
    for cat in blendshapes:
        name = cat.category_name
        if   name == "eyeBlinkLeft":  sl = cat.score
        elif name == "eyeBlinkRight": sr = cat.score

    if sl > threshold and sr > threshold and abs(sl - sr) < both_margin:
        return BlinkLabel.NONE          # simultaneous → ignore

    if sl > threshold:
        return BlinkLabel.LEFT
    if sr > threshold:
        return BlinkLabel.RIGHT
    return BlinkLabel.NONE


# ── helpers ───────────────────────────────────────────────────────────────────

def _tip_above(lm: List[Landmark], tip: int, pip: int) -> bool:
    return lm[tip].y < lm[pip].y


def _dist(lm: List[Landmark], a: int, b: int) -> float:
    dx = lm[a].x - lm[b].x
    dy = lm[a].y - lm[b].y
    return math.hypot(dx, dy)

# -*- coding: utf-8 -*-
"""
photo_editor.py
===============
Gesture-driven photo editor (OpenCV + Pillow).

Gestures
--------
  - Palm rotation              -> rotate image
  - Thumb/index spread change  -> zoom in/out
  - Horizontal swipe           -> switch filters
"""
from __future__ import annotations

import argparse
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from config import AppConfig
from core.tracker import Landmark, Tracker, open_camera
from ui.overlay import draw_hand


FILTERS = ("original", "bw", "sepia", "blur")


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _short_angle_diff(prev_deg: float, now_deg: float) -> float:
    """Smallest signed angle delta in degrees, in range [-180..180]."""
    d = (now_deg - prev_deg + 180.0) % 360.0 - 180.0
    return d


def _dist(lm: list[Landmark], a: int, b: int) -> float:
    dx = lm[a].x - lm[b].x
    dy = lm[a].y - lm[b].y
    return math.hypot(dx, dy)


def _tip_above(lm: list[Landmark], tip: int, pip: int) -> bool:
    return lm[tip].y < lm[pip].y


def _thumb_open(lm: list[Landmark], side: str) -> bool:
    return lm[4].x < lm[3].x if side == "Right" else lm[4].x > lm[3].x


@dataclass
class GestureDelta:
    rotate_deg: float = 0.0
    zoom_delta: float = 0.0
    swipe_dir: int = 0  # -1 left, +1 right
    open_score: int = 0


class PhotoGestureEngine:
    """
    Robust gesture deltas with:
      - low-pass filtering
      - deadzones
      - swipe cooldown
    """

    def __init__(self) -> None:
        self._last_palm_angle: Optional[float] = None
        self._rot_ema = 0.0

        self._last_spread: Optional[float] = None
        self._zoom_ema = 0.0

        self._wrist_x_hist: Deque[Tuple[float, float]] = deque(maxlen=7)
        self._last_swipe_ts = 0.0

        # Tuned defaults for stable interaction on typical webcams
        self.rot_alpha = 0.40
        self.rot_deadzone_deg = 0.40
        self.rot_gain = 1.20
        self.rot_max_step = 4.0

        self.zoom_alpha = 0.35
        self.zoom_deadzone = 0.0020
        self.zoom_gain = 2.20
        self.zoom_max_step = 0.045

        self.swipe_speed = 0.95
        self.swipe_cooldown = 0.80

    def update(self, lm: list[Landmark], side: str, now: float) -> GestureDelta:
        idx = _tip_above(lm, 8, 6)
        mid = _tip_above(lm, 12, 10)
        ring = _tip_above(lm, 16, 14)
        pink = _tip_above(lm, 20, 18)
        thumb = _thumb_open(lm, side)
        open_score = int(idx) + int(mid) + int(ring) + int(pink) + int(thumb)
        is_open = open_score >= 3

        # Palm orientation from index-mcp to pinky-mcp axis
        palm_angle = math.degrees(
            math.atan2(lm[5].y - lm[17].y, lm[5].x - lm[17].x)
        )

        rotate_deg = 0.0
        if self._last_palm_angle is not None and is_open:
            raw = _short_angle_diff(self._last_palm_angle, palm_angle)
            self._rot_ema = self.rot_alpha * raw + (1.0 - self.rot_alpha) * self._rot_ema
            if abs(self._rot_ema) > self.rot_deadzone_deg:
                rotate_deg = _clamp(
                    self._rot_ema * self.rot_gain, -self.rot_max_step, self.rot_max_step
                )
        self._last_palm_angle = palm_angle

        spread = _dist(lm, 4, 8)
        zoom_delta = 0.0
        if self._last_spread is not None and (idx or is_open):
            raw = spread - self._last_spread
            self._zoom_ema = self.zoom_alpha * raw + (1.0 - self.zoom_alpha) * self._zoom_ema
            if abs(self._zoom_ema) > self.zoom_deadzone:
                zoom_delta = _clamp(
                    self._zoom_ema * self.zoom_gain,
                    -self.zoom_max_step,
                    self.zoom_max_step,
                )
        self._last_spread = spread

        swipe_dir = 0
        self._wrist_x_hist.append((now, lm[0].x))
        if (
            len(self._wrist_x_hist) >= 4
            and is_open
            and (now - self._last_swipe_ts) > self.swipe_cooldown
        ):
            t0, x0 = self._wrist_x_hist[0]
            t1, x1 = self._wrist_x_hist[-1]
            dt = max(t1 - t0, 1e-3)
            vx = (x1 - x0) / dt
            if abs(vx) > self.swipe_speed:
                swipe_dir = 1 if vx > 0 else -1
                self._last_swipe_ts = now
                self._wrist_x_hist.clear()

        return GestureDelta(
            rotate_deg=rotate_deg,
            zoom_delta=zoom_delta,
            swipe_dir=swipe_dir,
            open_score=open_score,
        )


def _default_canvas() -> np.ndarray:
    h, w = 720, 1080
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    r = (40 + 180 * xx).astype(np.uint8)
    g = (30 + 150 * (1.0 - yy)).astype(np.uint8)
    b = (60 + 130 * (0.5 + 0.5 * np.sin(xx * 4.2))).astype(np.uint8)
    img = np.dstack([r, g, b])
    return img


def _load_rgb(path: str) -> np.ndarray:
    if path and os.path.exists(path):
        pil = Image.open(path).convert("RGB")
        return np.asarray(pil, dtype=np.uint8)
    return _default_canvas()


def _fit_max_side(rgb: np.ndarray, max_side: int = 1280) -> np.ndarray:
    h, w = rgb.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return rgb
    k = max_side / float(m)
    nw, nh = int(round(w * k)), int(round(h * k))
    return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)


def _apply_filter(rgb: np.ndarray, name: str) -> np.ndarray:
    pil = Image.fromarray(rgb, mode="RGB")
    if name == "bw":
        pil = ImageOps.grayscale(pil).convert("RGB")
    elif name == "sepia":
        gray = ImageOps.grayscale(pil)
        pil = ImageOps.colorize(gray, black="#25170F", white="#F4E3C1")
    elif name == "blur":
        pil = pil.filter(ImageFilter.GaussianBlur(radius=2.2))
    return np.asarray(pil, dtype=np.uint8)


def _apply_transform(rgb: np.ndarray, rotation_deg: float, scale: float) -> np.ndarray:
    h, w = rgb.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    mat = cv2.getRotationMatrix2D((cx, cy), rotation_deg, scale)
    out = cv2.warpAffine(
        rgb,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return out


class PhotoEditorApp:
    def __init__(self, image_path: str = "") -> None:
        self.cfg = AppConfig.load()
        self.base_rgb = _fit_max_side(_load_rgb(image_path))
        self.filter_idx = 0
        self.rotation_deg = 0.0
        self.scale = 1.0
        self.last_status = "Show palm to start gesture control"
        self._status_until = 0.0

        self.tracker = Tracker(self.cfg)
        self.engine = PhotoGestureEngine()

    def _status(self, msg: str, sec: float = 1.1) -> None:
        self.last_status = msg
        self._status_until = time.monotonic() + sec

    def _render_editor_frame(self) -> np.ndarray:
        rgb = _apply_filter(self.base_rgb, FILTERS[self.filter_idx])
        rgb = _apply_transform(rgb, self.rotation_deg, self.scale)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]

        # UI overlay
        cv2.rectangle(bgr, (0, 0), (w, 80), (12, 12, 12), -1)
        cv2.line(bgr, (0, 80), (w, 80), (80, 180, 220), 1)
        cv2.putText(
            bgr,
            f"Filter: {FILTERS[self.filter_idx].upper()}",
            (16, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            (100, 220, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            bgr,
            f"Rotation: {self.rotation_deg:.1f} deg   Zoom: {self.scale:.2f}x",
            (16, 58),
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            (185, 220, 185),
            1,
            cv2.LINE_AA,
        )
        now = time.monotonic()
        if now < self._status_until:
            cv2.putText(
                bgr,
                self.last_status,
                (w - 520, 58),
                cv2.FONT_HERSHEY_DUPLEX,
                0.50,
                (230, 200, 120),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            bgr,
            "Keys: Q/Esc quit | R reset | S save | [ / ] filter",
            (16, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        return bgr

    def _pick_main_hand(self, hands):
        if not hands:
            return None
        # Prefer configured cursor hand for stable behavior with two hands.
        pref = "Right" if self.cfg.handedness == "right" else "Left"
        for h in hands:
            if h.handedness == pref:
                return h
        return hands[0]

    def run(self) -> None:
        cap = open_camera(self.cfg)
        cv2.namedWindow("Gesture Photo Editor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gesture Photo Editor", 1100, 760)

        start = time.time()
        try:
            while True:
                ok, cam = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                cam = cv2.flip(cam, 1)

                ts_ms = int((time.time() - start) * 1000)
                result = self.tracker.detect(cam, ts_ms)
                hand = self._pick_main_hand(result.hands)

                if hand is not None:
                    draw_hand(cam, hand.landmarks, hand.handedness)
                    d = self.engine.update(
                        hand.landmarks, hand.handedness, time.monotonic()
                    )
                    self.rotation_deg = (self.rotation_deg + d.rotate_deg) % 360.0
                    # Multiplicative zoom feels more natural than additive.
                    self.scale = _clamp(self.scale * (1.0 + d.zoom_delta), 0.25, 3.50)
                    if d.swipe_dir != 0:
                        self.filter_idx = (self.filter_idx + d.swipe_dir) % len(FILTERS)
                        self._status(f"Filter -> {FILTERS[self.filter_idx].upper()}")

                frame = self._render_editor_frame()

                # Camera preview inset
                ih, iw = frame.shape[:2]
                cam_small = cv2.resize(cam, (320, 200), interpolation=cv2.INTER_AREA)
                y0, x0 = 92, iw - 336
                frame[y0:y0 + 200, x0:x0 + 320] = cam_small
                cv2.rectangle(frame, (x0 - 1, y0 - 1), (x0 + 320, y0 + 200), (90, 90, 90), 1)
                cv2.putText(
                    frame,
                    "Live camera",
                    (x0, y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.47,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow("Gesture Photo Editor", frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("r"), ord("R")):
                    self.rotation_deg = 0.0
                    self.scale = 1.0
                    self._status("Transform reset")
                elif key == ord("["):
                    self.filter_idx = (self.filter_idx - 1) % len(FILTERS)
                    self._status(f"Filter -> {FILTERS[self.filter_idx].upper()}")
                elif key == ord("]"):
                    self.filter_idx = (self.filter_idx + 1) % len(FILTERS)
                    self._status(f"Filter -> {FILTERS[self.filter_idx].upper()}")
                elif key in (ord("s"), ord("S")):
                    rgb = _apply_filter(self.base_rgb, FILTERS[self.filter_idx])
                    rgb = _apply_transform(rgb, self.rotation_deg, self.scale)
                    out = f"edited_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    Image.fromarray(rgb, mode="RGB").save(out)
                    self._status(f"Saved: {out}", 1.8)

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.tracker.close()


def run_photo_editor(image_path: str = "") -> None:
    PhotoEditorApp(image_path=image_path).run()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gesture Photo Editor (OpenCV + Pillow)")
    p.add_argument(
        "--image",
        default="",
        help="Path to source image. If omitted, demo canvas is used.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_photo_editor(image_path=args.image)

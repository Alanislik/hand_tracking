# -*- coding: utf-8 -*-
"""config.py — Single source of truth for every tuneable parameter."""
from __future__ import annotations
import json, os
from dataclasses import dataclass, field, asdict
from typing import List

CFG_PATH = os.path.expanduser("~/.gesture_v8.json")


@dataclass
class CameraConfig:
    indices: List[int] = field(default_factory=lambda: [0, 1, 2])
    width:   int   = 640
    height:  int   = 480
    fps:     int   = 30
    buffer:  int   = 1
    dshow:   bool  = True


@dataclass
class TrackingConfig:
    max_hands:          int   = 2
    hand_detect_conf:   float = 0.60
    hand_presence_conf: float = 0.60
    hand_track_conf:    float = 0.55
    face_detect_conf:   float = 0.60
    face_presence_conf: float = 0.60
    face_track_conf:    float = 0.55


@dataclass
class SmoothingConfig:
    ema_window:      int   = 8
    deadzone_px:     int   = 4
    velocity_cap:    int   = 120
    accel_exponent:  float = 1.0   # kept for compatibility; smoother ignores it
    # Scroll-specific low-pass filter (0 < α ≤ 1; smaller = smoother)
    scroll_alpha:    float = 0.25
    # Minimum scroll delta to actually fire a wheel event (normalised units)
    scroll_deadzone: float = 0.004


@dataclass
class CalibConfig:
    zone: List[float] = field(
        default_factory=lambda: [0.15, 0.10, 0.85, 0.85]
    )


@dataclass
class GestureConfig:
    pinch_idx_thr:   float = 0.048
    pinch_mid_thr:   float = 0.060
    dclick_gap:      float = 0.36
    # Hold a pinch longer than this (sec) → drag instead of click
    drag_hold_sec:   float = 0.40
    rclick_cooldown: float = 0.75
    scroll_speed:    int   = 120
    kb_cooldown:     float = 1.5


@dataclass
class BlinkConfig:
    threshold:   float = 0.40
    min_frames:  int   = 2
    cooldown:    float = 0.60
    both_margin: float = 0.12


@dataclass
class AppConfig:
    camera:    CameraConfig    = field(default_factory=CameraConfig)
    tracking:  TrackingConfig  = field(default_factory=TrackingConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    calib:     CalibConfig     = field(default_factory=CalibConfig)
    gesture:   GestureConfig   = field(default_factory=GestureConfig)
    blink:     BlinkConfig     = field(default_factory=BlinkConfig)
    mode:      str             = "hand"   # "hand" | "head"
    sensitivity: float         = 1.0
    # "right" → right hand moves cursor, left hand acts (default)
    # "left"  → left hand moves cursor, right hand acts
    handedness:  str           = "right"

    def save(self) -> None:
        with open(CFG_PATH, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load() -> "AppConfig":
        if not os.path.exists(CFG_PATH):
            return AppConfig()
        try:
            with open(CFG_PATH, encoding="utf-8") as f:
                d = json.load(f)
            c = AppConfig()
            for key, cls in [
                ("camera",    CameraConfig),
                ("tracking",  TrackingConfig),
                ("smoothing", SmoothingConfig),
                ("calib",     CalibConfig),
                ("gesture",   GestureConfig),
                ("blink",     BlinkConfig),
            ]:
                if key in d:
                    setattr(c, key, cls(**{
                        k: v for k, v in d[key].items()
                        if k in cls.__dataclass_fields__
                    }))
            if "mode"        in d: c.mode        = d["mode"]
            if "sensitivity" in d: c.sensitivity = d["sensitivity"]
            if "handedness"  in d: c.handedness  = d["handedness"]
            return c
        except Exception:
            return AppConfig()

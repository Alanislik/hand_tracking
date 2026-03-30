# -*- coding: utf-8 -*-
"""
core/state_machine.py
=====================
GestureLabel → debounced Event stream.

States (HandStateMachine)
-------------------------
IDLE          no active gesture
MOVING        index up → cursor follows lm[8]
PINCH_HOLD    pinch entered; waiting to decide click vs drag
DRAGGING      pinch held > drag_hold_sec → mouse button held down
PINCHED_IDX   pinch released before drag threshold → click emitted
PINCHED_MID   thumb+middle pinch
SCROLLING     two-finger scroll
OPEN_PALM     keyboard toggle held

Events emitted
--------------
  move        nx, ny
  lclick
  dclick
  rclick
  mouse_down  (drag start)
  mouse_up    (drag end)
  scroll      scroll_dy
  keyboard
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, List

from core.gestures import GestureLabel, BlinkLabel
from config import GestureConfig, BlinkConfig, SmoothingConfig


@dataclass
class Event:
    kind:      str
    scroll_dy: int   = 0
    nx:        float = 0.0
    ny:        float = 0.0


class HandStateMachine:
    """One instance per hand."""

    def __init__(self, cfg: GestureConfig, scfg: SmoothingConfig) -> None:
        self._cfg           = cfg
        self._scfg          = scfg
        self._state         = "IDLE"
        self._last_lclick   = 0.0
        self._kb_cooldown   = 0.0
        self._scroll_ref_y: Optional[float] = None

        # Drag: timestamp when pinch first entered
        self._pinch_start: Optional[float] = None

        # Scroll low-pass state (IIR filter)
        self._scroll_vel: float = 0.0  # filtered delta

    @property
    def state(self) -> str:
        return self._state

    def update(self, label: GestureLabel,
               lm_tip_x: float = 0.0,
               lm_tip_y: float = 0.0) -> List[Event]:
        events: List[Event] = []
        now    = time.monotonic()
        cfg    = self._cfg

        # ── MOVE ─────────────────────────────────────────────
        if label == GestureLabel.MOVE:
            if self._state == "DRAGGING":
                # Finger released while dragging → mouse_up
                events.append(Event("mouse_up"))
            self._state        = "MOVING"
            self._scroll_ref_y = None
            self._pinch_start  = None
            self._scroll_vel   = 0.0
            events.append(Event("move", nx=lm_tip_x, ny=lm_tip_y))

        # ── PINCH INDEX (click / drag) ────────────────────────
        elif label == GestureLabel.PINCH_IDX:
            if self._state == "DRAGGING":
                # Still dragging — keep emitting move so cursor follows
                events.append(Event("move", nx=lm_tip_x, ny=lm_tip_y))

            elif self._state not in ("PINCH_HOLD", "PINCHED_IDX"):
                # Fresh pinch entry
                self._pinch_start = now
                self._state = "PINCH_HOLD"

            elif self._state == "PINCH_HOLD":
                held = now - (self._pinch_start or now)
                if held >= cfg.drag_hold_sec:
                    # Held long enough → start drag
                    events.append(Event("mouse_down"))
                    self._state = "DRAGGING"
                # else: still waiting — no event yet

        # ── RELEASE from pinch ────────────────────────────────
        # (any non-pinch label when we were in PINCH_HOLD = click)
        else:
            if self._state == "PINCH_HOLD":
                # Released before drag threshold → fire click
                dt = now - self._last_lclick
                if dt < cfg.dclick_gap:
                    events.append(Event("dclick"))
                    self._last_lclick = 0.0
                else:
                    events.append(Event("lclick"))
                    self._last_lclick = now
                self._state = "IDLE"

            elif self._state == "DRAGGING":
                events.append(Event("mouse_up"))
                self._state = "IDLE"

            # ── PINCH MID ────────────────────────────────────
            if label == GestureLabel.PINCH_MID:
                if self._state not in ("PINCHED_MID",):
                    if now - self._last_lclick > cfg.rclick_cooldown:
                        events.append(Event("rclick"))
                        self._last_lclick = now
                    self._state = "PINCHED_MID"

            # ── SCROLL ───────────────────────────────────────
            elif label == GestureLabel.SCROLL:
                self._state = "SCROLLING"
                if self._scroll_ref_y is None:
                    self._scroll_ref_y = lm_tip_y
                    self._scroll_vel   = 0.0
                else:
                    raw_dy = (self._scroll_ref_y - lm_tip_y)
                    # Low-pass IIR: v = α·raw + (1-α)·v_prev
                    α = self._scfg.scroll_alpha
                    self._scroll_vel = α * raw_dy + (1 - α) * self._scroll_vel
                    # Dead-zone: only fire if magnitude > threshold
                    if abs(self._scroll_vel) > self._scfg.scroll_deadzone:
                        dy = int(self._scroll_vel * 800)
                        if dy != 0:
                            events.append(Event("scroll", scroll_dy=dy))

            # ── OPEN PALM ─────────────────────────────────────
            elif label == GestureLabel.OPEN_PALM:
                if self._state != "OPEN_PALM":
                    if now > self._kb_cooldown:
                        events.append(Event("keyboard"))
                        self._kb_cooldown = now + cfg.kb_cooldown
                    self._state = "OPEN_PALM"

            # ── FIST / IDLE / NONE ────────────────────────────
            elif label in (GestureLabel.FIST, GestureLabel.IDLE, GestureLabel.NONE):
                self._state        = "IDLE"
                self._scroll_ref_y = None
                self._scroll_vel   = 0.0
                self._pinch_start  = None

        return events


class BlinkStateMachine:
    def __init__(self, cfg: BlinkConfig) -> None:
        self._cfg   = cfg
        self._cnt_l = 0; self._cnt_r = 0
        self._last_l = 0.0; self._last_r = 0.0

    def update(self, label: BlinkLabel) -> List[Event]:
        events: List[Event] = []
        now = time.monotonic()
        cfg = self._cfg

        if label == BlinkLabel.NONE:
            self._cnt_l = 0; self._cnt_r = 0
            return events

        if label == BlinkLabel.LEFT:
            self._cnt_l += 1; self._cnt_r = 0
            if self._cnt_l >= cfg.min_frames and now - self._last_l > cfg.cooldown:
                events.append(Event("lclick"))
                self._last_l = now; self._cnt_l = 0

        elif label == BlinkLabel.RIGHT:
            self._cnt_r += 1; self._cnt_l = 0
            if self._cnt_r >= cfg.min_frames and now - self._last_r > cfg.cooldown:
                events.append(Event("rclick"))
                self._last_r = now; self._cnt_r = 0

        return events

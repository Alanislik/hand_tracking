# -*- coding: utf-8 -*-
"""
main.py — GestureControl v8.1
==============================
Changes vs v8:
  • Left-handed mode  (cfg.handedness = "right"|"left")
  • Drag & drop       (mouse_down / mouse_up via state machine)
  • Scroll smoothing  (IIR low-pass in HandStateMachine)
  • Cursor stability  (accel threshold in CursorSmoother)
  • Strategy pattern  input_controller (BaseInputController hierarchy)
  • Key 'L' toggles handedness live
"""
from __future__ import annotations
import os, sys, time, threading
import argparse
from collections import deque
from typing import Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel",      "3")

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config              import AppConfig
from core.tracker        import Tracker, open_camera
from core.gestures       import classify_hand, classify_blink, GestureLabel, BlinkLabel
from core.state_machine  import HandStateMachine, BlinkStateMachine, Event
from core.mapper         import CoordinateMapper
from core.smoothing      import CursorSmoother
from input.input_controller import InputController, BaseInputController
from ui.overlay          import HUDState, draw_frame, draw_hand, draw_face_dots
from photo_editor        import run_photo_editor


class Pipeline:
    def __init__(self, cfg: AppConfig, ctrl: BaseInputController) -> None:
        sw, sh       = ctrl.screen_size()
        self.cfg     = cfg
        self.ctrl    = ctrl
        self.tracker = Tracker(cfg)
        self.mapper  = CoordinateMapper(cfg.calib, sw, sh, cfg.sensitivity)
        self.smoother= CursorSmoother(cfg.smoothing, sw, sh)
        # Each hand gets its own state machine; roles are swapped by App
        self.sm_r    = HandStateMachine(cfg.gesture, cfg.smoothing)
        self.sm_l    = HandStateMachine(cfg.gesture, cfg.smoothing)
        self.sm_blink= BlinkStateMachine(cfg.blink)
        self.calib_pts: list = []

    def reset_smoothing(self) -> None:
        self.smoother.clear()


class App:
    def __init__(self) -> None:
        self.cfg  = AppConfig.load()
        self.ctrl = InputController()
        self.pipe = Pipeline(self.cfg, self.ctrl)

        self._lock  = threading.Lock()
        self._fbuf: Optional[np.ndarray] = None
        self._stop  = threading.Event()

        self.hud = HUDState()
        self.hud.platform_tag = "WIN" if sys.platform == "win32" else "LNX"
        self.hud.mode         = self.cfg.mode
        self.hud.sensitivity  = self.cfg.sensitivity
        self.hud.smoothing    = self.cfg.smoothing.ema_window

        self.calibrating = False
        self.calib_step  = 0
        self.last_tip: Optional[Tuple[float, float]] = None

        self.fps_buf = deque(maxlen=60)
        self.last_t  = time.perf_counter()

        # Drag state: track whether button is currently held
        self._dragging = False

    # ── helpers ───────────────────────────────────────────────

    def _status(self, msg: str, dur: float = 2.0) -> None:
        self.hud.status_msg = msg
        self.hud.status_end = time.monotonic() + dur

    def _flash(self) -> None:
        self.hud.click_flash_end = time.monotonic() + 0.25

    # ── hand roles (respects handedness setting) ──────────────

    @property
    def _cursor_hand(self) -> str:
        """Which MediaPipe label drives the cursor."""
        return "Right" if self.cfg.handedness == "right" else "Left"

    @property
    def _action_hand(self) -> str:
        """Which MediaPipe label drives clicks/scroll."""
        return "Left" if self.cfg.handedness == "right" else "Right"

    # ── event dispatcher ──────────────────────────────────────

    def _dispatch(self, events: list) -> None:
        for ev in events:
            if ev.kind == "move":
                sx, sy = self.pipe.mapper.map(ev.nx, ev.ny)
                mx, my = self.pipe.smoother.push(sx, sy)
                self.ctrl.move(mx, my)

            elif ev.kind == "lclick":
                self.ctrl.click("left"); self._flash()
                self._status("Left Click", .4)

            elif ev.kind == "dclick":
                self.ctrl.double_click(); self._flash()
                self._status("Double Click", .4)

            elif ev.kind == "rclick":
                self.ctrl.click("right")
                self._status("Right Click", .4)

            elif ev.kind == "mouse_down":
                self.ctrl.mouse_down("left")
                self._dragging = True
                self._flash()
                self._status("Dragging…", 99.0)

            elif ev.kind == "mouse_up":
                self.ctrl.mouse_up("left")
                self._dragging = False
                self._status("Drop", .4)

            elif ev.kind == "scroll":
                self.ctrl.scroll(ev.scroll_dy)

            elif ev.kind == "keyboard":
                threading.Thread(
                    target=self.ctrl.toggle_keyboard, daemon=True
                ).start()
                self._status("Keyboard toggled", 1.0)

    # ── tracking thread ───────────────────────────────────────

    def _tracking_loop(self, cap: cv2.VideoCapture) -> None:
        BUDGET = 1.0 / 30.0
        t0     = time.time()
        pipe   = self.pipe

        while not self._stop.is_set():
            t_s = time.perf_counter()

            ret, raw = cap.read()
            if not ret:
                time.sleep(0.005); continue

            raw   = cv2.flip(raw, 1)
            ts_ms = int((time.time() - t0) * 1000)
            out   = raw.copy()

            try:
                result = pipe.tracker.detect(raw, ts_ms)

                if self.cfg.mode == "hand":
                    lm_map = {h.handedness: h for h in result.hands}

                    for hand in result.hands:
                        draw_hand(out, hand.landmarks, hand.handedness)

                    # Cursor hand
                    ch = self._cursor_hand
                    if ch in lm_map:
                        hand = lm_map[ch]
                        lm   = hand.landmarks
                        tip  = lm[8]
                        self.last_tip = (tip.x, tip.y)

                        g  = classify_hand(lm, ch,
                                           self.cfg.gesture.pinch_idx_thr,
                                           self.cfg.gesture.pinch_mid_thr)
                        ev = pipe.sm_r.update(g, tip.x, tip.y) \
                            if not self.calibrating else []

                        self.hud.gesture_r = g.value
                        self.hud.state_r   = pipe.sm_r.state
                        self._dispatch(ev)
                    else:
                        self.hud.gesture_r = "none"
                        self.hud.state_r   = "IDLE"

                    # Action hand — clicks/scroll only, never moves cursor
                    ah = self._action_hand
                    if ah in lm_map:
                        hand = lm_map[ah]
                        lm   = hand.landmarks
                        tip  = lm[8]

                        g  = classify_hand(lm, ah,
                                           self.cfg.gesture.pinch_idx_thr,
                                           self.cfg.gesture.pinch_mid_thr)
                        ev = pipe.sm_l.update(g, tip.x, tip.y) \
                            if not self.calibrating else []

                        self.hud.gesture_l = g.value
                        self.hud.state_l   = pipe.sm_l.state
                        # Filter out "move" — action hand never controls cursor
                        self._dispatch([e for e in ev if e.kind != "move"])
                    else:
                        self.hud.gesture_l = "none"
                        self.hud.state_l   = "IDLE"

                else:  # head mode
                    if result.face:
                        fl = result.face.landmarks
                        draw_face_dots(out, fl)
                        if not self.calibrating:
                            nose   = fl[1]
                            sx, sy = pipe.mapper.map(nose.x, nose.y)
                            mx, my = pipe.smoother.push(sx, sy)
                            self.ctrl.move(mx, my)
                        if result.face.blendshapes:
                            bl  = classify_blink(result.face.blendshapes,
                                                 self.cfg.blink.threshold,
                                                 self.cfg.blink.both_margin)
                            evs = pipe.sm_blink.update(bl)
                            self.hud.face_label = bl.value if bl != BlinkLabel.NONE else ""
                            self._dispatch(evs)

            except Exception as exc:
                print(f"[WARN] {exc}")

            # FPS
            now_p = time.perf_counter()
            self.fps_buf.append(1.0 / max(now_p - self.last_t, 1e-6))
            self.last_t      = now_p
            self.hud.fps     = float(np.mean(self.fps_buf))
            self.hud.calib_zone  = list(self.cfg.calib.zone)
            self.hud.calibrating = self.calibrating
            self.hud.calib_step  = self.calib_step
            self.hud.calib_pt0   = (pipe.calib_pts[0] if pipe.calib_pts else None)

            with self._lock:
                self._fbuf = draw_frame(out, self.hud)

            sl = BUDGET - (time.perf_counter() - t_s)
            if sl > 0: time.sleep(sl)

    # ── calibration ───────────────────────────────────────────

    def _calib_key(self, key: int) -> None:
        if key == 27:
            self.calibrating = False; self.pipe.calib_pts = []
            self._status("Calibration cancelled")
        elif key == 32:
            if self.last_tip is None:
                self._status("No hand detected!", 1.5); return
            self.pipe.calib_pts.append(self.last_tip)
            if self.calib_step == 0:
                self.calib_step = 1
                self._status("Top-Left saved.  Now Bottom-Right", 2.5)
            else:
                p0, p1 = self.pipe.calib_pts
                self.cfg.calib.zone = [
                    min(p0[0],p1[0]), min(p0[1],p1[1]),
                    max(p0[0],p1[0]), max(p0[1],p1[1]),
                ]
                self.pipe.mapper.update_calib(self.cfg.calib.zone)
                self.calibrating = False; self.pipe.calib_pts = []
                self.cfg.save(); self._status("Calibration saved!", 2.5)

    # ── key handler ───────────────────────────────────────────

    def _handle_key(self, key: int) -> bool:
        if key in (ord('q'), ord('Q'), 27): return True

        elif key in (ord('h'), ord('H')):
            self.hud.show_help = not self.hud.show_help

        elif key in (ord('b'), ord('B')):
            self.hud.debug = not self.hud.debug

        elif key in (ord('l'), ord('L')):
            # Toggle left-handed mode
            self.cfg.handedness = "left" if self.cfg.handedness == "right" else "right"
            self.pipe.sm_r = HandStateMachine(self.cfg.gesture, self.cfg.smoothing)
            self.pipe.sm_l = HandStateMachine(self.cfg.gesture, self.cfg.smoothing)
            self.pipe.reset_smoothing()
            hand = "LEFT" if self.cfg.handedness == "left" else "RIGHT"
            self._status(f"Cursor hand: {hand}", 2.0)

        elif key in (ord('c'), ord('C')):
            self.calibrating = True; self.calib_step = 0
            self.pipe.calib_pts = []; self.last_tip = None
            self._status(f"Calibrate — show {self._cursor_hand} index finger", 2.0)

        elif key in (ord('r'), ord('R')):
            self.cfg.calib.zone = [0.15, 0.10, 0.85, 0.85]
            self.pipe.mapper.update_calib(self.cfg.calib.zone)
            self._status("Calibration reset")

        elif key in (ord('m'), ord('M')):
            self.cfg.mode = "head" if self.cfg.mode == "hand" else "hand"
            self.hud.mode = self.cfg.mode
            self.pipe.reset_smoothing()
            self._status(f"Mode: {self.cfg.mode.upper()}")

        elif key in (ord('d'), ord('D')):
            self.cfg.sensitivity = min(5.0, round(self.cfg.sensitivity + 0.1, 1))
            self.pipe.mapper.sensitivity = self.cfg.sensitivity
            self.hud.sensitivity = self.cfg.sensitivity
            self._status(f"Sensitivity: {self.cfg.sensitivity:.1f}")

        elif key in (ord('a'), ord('A')):
            self.cfg.sensitivity = max(0.2, round(self.cfg.sensitivity - 0.1, 1))
            self.pipe.mapper.sensitivity = self.cfg.sensitivity
            self.hud.sensitivity = self.cfg.sensitivity
            self._status(f"Sensitivity: {self.cfg.sensitivity:.1f}")

        elif key in (ord('w'), ord('W')):
            self.cfg.smoothing.ema_window = min(25, self.cfg.smoothing.ema_window + 1)
            self.pipe.smoother.resize(self.cfg.smoothing.ema_window)
            self.hud.smoothing = self.cfg.smoothing.ema_window
            self._status(f"Smoothing: {self.cfg.smoothing.ema_window}")

        elif key in (ord('s'), ord('S')):
            self.cfg.smoothing.ema_window = max(1, self.cfg.smoothing.ema_window - 1)
            self.pipe.smoother.resize(self.cfg.smoothing.ema_window)
            self.hud.smoothing = self.cfg.smoothing.ema_window
            self._status(f"Smoothing: {self.cfg.smoothing.ema_window}")

        elif key in (ord('p'), ord('P')):
            self.cfg.save(); self._status("Config saved!")

        return False

    # ── run ───────────────────────────────────────────────────

    def run(self) -> None:
        cap = open_camera(self.cfg)

        # Keep saved config between runs. If you need factory defaults,
        # delete ~/.gesture_v8.json manually or add a dedicated reset key.

        sw, sh = self.ctrl.screen_size()
        z = self.cfg.calib.zone
        print("GestureControl v8.1")
        print(f"Screen : {sw}x{sh}")
        print(f"Zone   : x=[{z[0]:.2f}..{z[2]:.2f}]  y=[{z[1]:.2f}..{z[3]:.2f}]  "
              f"(C to calibrate, R to reset)")
        print(f"Sens   : {self.cfg.sensitivity}  (A/D to change)")
        print("H=help  B=debug  L=handedness  C=calib  R=reset  M=mode  "
              "A/D=sens  S/W=smooth  P=save  Q=quit")
        if sys.platform == "win32":
            print("TIP: Run as Administrator for elevated windows.")
        print()
        print(">>> If cursor runs off screen: press C and calibrate your zone! <<<")

        cv2.namedWindow("GestureControl", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("GestureControl", 640, 480)

        tracker = threading.Thread(
            target=self._tracking_loop, args=(cap,), daemon=True
        )
        tracker.start()

        try:
            while not self._stop.is_set():
                with self._lock:
                    frame = self._fbuf
                if frame is not None:
                    cv2.imshow("GestureControl", frame)
                key = cv2.waitKey(16) & 0xFF

                if self.calibrating:
                    if key != 255: self._calib_key(key)
                    continue
                if key == 255: continue
                if self._handle_key(key): break

        except KeyboardInterrupt:
            print("\n[INFO] Ctrl+C")
        finally:
            # Safety: release mouse button if we're mid-drag
            if self._dragging:
                try: self.ctrl.mouse_up("left")
                except Exception: pass
            self._stop.set()
            tracker.join(timeout=2.0)
            cap.release()
            cv2.destroyAllWindows()
            self.pipe.tracker.close()
            self.ctrl.close()
            self.cfg.save()
            print("[INFO] Done.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GestureControl launcher")
    p.add_argument(
        "--app",
        choices=("editor", "cursor"),
        default="editor",
        help="editor = gesture photo editor, cursor = legacy cursor controller",
    )
    p.add_argument(
        "--image",
        default="",
        help="Source image path for editor mode",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.app == "cursor":
        App().run()
    else:
        run_photo_editor(image_path=args.image)

# -*- coding: utf-8 -*-
"""
core/tracker.py
===============
Responsibilities (and ONLY these):
  • Download MediaPipe model files on first run
  • Open the camera (tries multiple indices, fails clearly)
  • Wrap HandLandmarker + FaceLandmarker
  • Return raw landmark results — zero business logic here

Output types are plain dataclasses so the rest of the pipeline
never imports mediapipe directly.
"""
from __future__ import annotations
import os, sys, time, urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Any

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from config import AppConfig

# ── model management ──────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

HAND_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
FACE_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")

_URLS = {
    HAND_PATH: ("https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"),
    FACE_PATH: ("https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"),
}


def ensure_models() -> None:
    for path, url in _URLS.items():
        if os.path.exists(path):
            continue
        name = os.path.basename(path)
        print(f"[DL] {name} …")

        def _prog(b, bs, t):
            pct = min(100, int(b * bs * 100 / t)) if t > 0 else 0
            sys.stdout.write(f"\r  {pct}%"); sys.stdout.flush()

        try:
            urllib.request.urlretrieve(url, path, reporthook=_prog)
            print(f"\n[OK] {name}")
        except Exception as exc:
            print(f"\n[ERR] {exc}\n  URL: {url}\n  Save as: {path}")
            raise SystemExit(1)


# ── output data types ─────────────────────────────────────────────────────────

@dataclass
class Landmark:
    """Normalised (0-1) position of a single point."""
    x: float
    y: float
    z: float = 0.0


@dataclass
class HandResult:
    """One detected hand."""
    landmarks:  List[Landmark]
    handedness: str              # "Left" or "Right" (already mirror-corrected)


@dataclass
class FaceResult:
    landmarks:   List[Landmark]
    blendshapes: List[Any]       # raw MediaPipe Category objects


@dataclass
class TrackerOutput:
    hands: List[HandResult]      = field(default_factory=list)
    face:  Optional[FaceResult]  = None


# ── camera ────────────────────────────────────────────────────────────────────

def open_camera(cfg: AppConfig) -> cv2.VideoCapture:
    backend = (cv2.CAP_DSHOW
               if cfg.camera.dshow and sys.platform == "win32"
               else cv2.CAP_ANY)

    for idx in cfg.camera.indices:
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
        cap.set(cv2.CAP_PROP_FPS,          cfg.camera.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   cfg.camera.buffer)
        ok, _ = cap.read()
        if ok:
            for _ in range(4): cap.read()   # flush stale frames
            print(f"[CAM] index={idx}")
            return cap
        cap.release()

    raise RuntimeError(
        f"No camera found. Tried indices: {cfg.camera.indices}"
    )


# ── MediaPipe wrappers ────────────────────────────────────────────────────────

class Tracker:
    """
    Wraps both HandLandmarker and FaceLandmarker.
    Call detect() every frame; returns TrackerOutput.
    """

    def __init__(self, cfg: AppConfig) -> None:
        t = cfg.tracking

        hand_opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=t.max_hands,
            min_hand_detection_confidence=t.hand_detect_conf,
            min_hand_presence_confidence=t.hand_presence_conf,
            min_tracking_confidence=t.hand_track_conf,
        )
        self._hands = mp_vision.HandLandmarker.create_from_options(hand_opts)

        face_opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=FACE_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=t.face_detect_conf,
            min_face_presence_confidence=t.face_presence_conf,
            min_tracking_confidence=t.face_track_conf,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self._face = mp_vision.FaceLandmarker.create_from_options(face_opts)

    def detect(self, frame_bgr, ts_ms: int) -> TrackerOutput:
        """
        frame_bgr : already-flipped BGR numpy array
        ts_ms     : monotonically increasing timestamp in milliseconds
        """
        import cv2 as _cv2
        rgb   = _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2RGB)
        mp_im = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # ── hands ──
        hand_res = self._hands.detect_for_video(mp_im, ts_ms)
        hands: List[HandResult] = []
        if hand_res.hand_landmarks:
            for lm_list, hness in zip(
                hand_res.hand_landmarks, hand_res.handedness
            ):
                raw   = hness[0].display_name          # "Left" / "Right"
                label = "Left" if raw == "Right" else "Right"  # mirror fix
                lms   = [Landmark(l.x, l.y, l.z) for l in lm_list]
                hands.append(HandResult(landmarks=lms, handedness=label))

        # Stable hand assignment by wrist x-position
        hands = _assign_hands(hands)

        # ── face ──
        face_res = self._face.detect_for_video(mp_im, ts_ms)
        face: Optional[FaceResult] = None
        if face_res.face_landmarks:
            lms  = [Landmark(l.x, l.y, l.z) for l in face_res.face_landmarks[0]]
            bsps = face_res.face_blendshapes[0] if face_res.face_blendshapes else []
            face = FaceResult(landmarks=lms, blendshapes=bsps)

        return TrackerOutput(hands=hands, face=face)

    def close(self) -> None:
        self._hands.close()
        self._face.close()


def _assign_hands(hands: List[HandResult]) -> List[HandResult]:
    """Re-assign Left/Right by wrist x-position (stable when hands are close)."""
    if len(hands) < 2:
        return hands
    # sort by wrist (lm[0]) x — smaller x = left side of (mirrored) frame
    hands.sort(key=lambda h: h.landmarks[0].x)
    hands[0].handedness = "Left"
    hands[1].handedness = "Right"
    return hands


ensure_models()

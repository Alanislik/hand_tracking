# -*- coding: utf-8 -*-
"""
ui/overlay.py — Everything visual.  Pure functions.  Zero side-effects.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

# ── skeleton constants ────────────────────────────────────────────────────────
HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]
TIPS = {4,8,12,16,20}
HAND_COL = {"Right":(0,200,255), "Left":(0,255,110)}
GEST_COL = {
    "move":(0,220,255),"pinch_idx":(0,255,70),"pinch_mid":(255,150,0),
    "scroll":(180,60,255),"fist":(140,140,140),
    "open_palm":(255,220,0),"idle":(60,60,60),"none":(60,60,60),
    # state machine event names for HUD
    "lclick":(0,255,70),"dclick":(0,255,70),"rclick":(255,150,0),
    "keyboard":(255,220,0),
}


# ── HUD snapshot (written by tracking thread, read by display thread) ─────────
@dataclass
class HUDState:
    fps:             float        = 0.0
    mode:            str          = "hand"
    sensitivity:     float        = 1.0
    smoothing:       int          = 8
    platform_tag:    str          = "?"
    calib_zone:      list         = field(default_factory=lambda:[.15,.10,.85,.85])
    gesture_r:       str          = "none"
    gesture_l:       str          = "none"
    state_r:         str          = "IDLE"
    state_l:         str          = "IDLE"
    face_label:      str          = ""
    status_msg:      str          = ""
    status_end:      float        = 0.0
    click_flash_end: float        = 0.0
    calibrating:     bool         = False
    calib_step:      int          = 0
    calib_pt0:       Optional[Tuple[float,float]] = None
    show_help:       bool         = True
    debug:           bool         = False


# ── main draw function ────────────────────────────────────────────────────────
def draw_frame(frame: np.ndarray, s: HUDState) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    now  = time.monotonic()

    # calib zone
    z  = s.calib_zone
    x0,y0 = int(z[0]*w),int(z[1]*h)
    x1,y1 = int(z[2]*w),int(z[3]*h)
    cv2.rectangle(out,(x0,y0),(x1,y1),(0,200,80),1)
    for px,py,dx,dy in [(x0,y0,1,1),(x1,y0,-1,1),(x0,y1,1,-1),(x1,y1,-1,-1)]:
        cv2.line(out,(px,py),(px+dx*12,py),(0,200,80),2)
        cv2.line(out,(px,py),(px,py+dy*12),(0,200,80),2)

    # click flash
    if s.click_flash_end > now:
        a=(s.click_flash_end-now)/0.25
        ov=out.copy(); cv2.rectangle(ov,(0,0),(w,h),(0,220,180),cv2.FILLED)
        cv2.addWeighted(ov,a*0.12,out,1-a*0.12,0,out)

    # top bar
    cv2.rectangle(out,(0,0),(w,42),(8,8,8),-1)
    cv2.line(out,(0,42),(w,42),(0,160,70),1)
    mc=(0,220,130) if s.mode=="hand" else (255,200,0)
    cv2.putText(out,f"[{s.platform_tag}] {s.mode.upper()}  FPS:{s.fps:.0f}",
                (10,27),cv2.FONT_HERSHEY_DUPLEX,.55,mc,1,cv2.LINE_AA)
    cv2.putText(out,f"SENS:{s.sensitivity:.1f}  SMTH:{s.smoothing}",
                (w-215,27),cv2.FONT_HERSHEY_DUPLEX,.50,(110,110,110),1,cv2.LINE_AA)

    # bottom gesture / state row
    if s.mode=="hand":
        _glabel(out,f"R:{s.gesture_r.upper()} [{s.state_r}]",10,h-10,s.gesture_r)
        _glabel(out,f"L:{s.gesture_l.upper()} [{s.state_l}]",w//2,h-10,s.gesture_l)
    elif s.face_label:
        cv2.putText(out,s.face_label,(10,h-10),cv2.FONT_HERSHEY_DUPLEX,
                    .5,(0,220,255),1,cv2.LINE_AA)

    # status message
    if now < s.status_end:
        fade=min(1.,(s.status_end-now)/.3)
        col=(int(70*fade),int(230*fade),int(150*fade))
        tw=cv2.getTextSize(s.status_msg,cv2.FONT_HERSHEY_DUPLEX,.75,2)[0][0]
        cv2.putText(out,s.status_msg,(w//2-tw//2,h//2),
                    cv2.FONT_HERSHEY_DUPLEX,.75,col,2,cv2.LINE_AA)

    # calibration overlay
    if s.calibrating:
        ov=out.copy(); cv2.rectangle(ov,(0,0),(w,h),(0,0,0),-1)
        cv2.addWeighted(ov,.55,out,.45,0,out)
        msgs=[("CALIBRATION",(0,230,130),.8,2),
              (("Step 1: right index → TOP-LEFT, press SPACE" if s.calib_step==0
                else "Step 2: right index → BOTTOM-RIGHT, press SPACE"),
               (255,235,70),.52,1),
              ("ESC = cancel",(130,130,130),.45,1)]
        for i,(txt,col,sz,th) in enumerate(msgs):
            tw=cv2.getTextSize(txt,cv2.FONT_HERSHEY_DUPLEX,sz,th)[0][0]
            cv2.putText(out,txt,(w//2-tw//2,h//2-24+i*36),
                        cv2.FONT_HERSHEY_DUPLEX,sz,col,th,cv2.LINE_AA)
        if s.calib_pt0:
            px,py=int(s.calib_pt0[0]*w),int(s.calib_pt0[1]*h)
            cv2.drawMarker(out,(px,py),(0,230,130),cv2.MARKER_CROSS,22,2)

    # help
    if s.show_help:
        _draw_help(out,w)

    # debug
    if s.debug:
        _draw_debug(out,s)

    return out


def draw_hand(out: np.ndarray, landmarks, label: str) -> None:
    h,w=out.shape[:2]
    col=HAND_COL.get(label,(200,200,200))
    pts=[(int(l.x*w),int(l.y*h)) for l in landmarks]
    for a,b in HAND_CONN: cv2.line(out,pts[a],pts[b],col,2,cv2.LINE_AA)
    for i,(px,py) in enumerate(pts):
        cv2.circle(out,(px,py),5 if i in TIPS else 3,col,-1,cv2.LINE_AA)
    cv2.putText(out,label,(pts[0][0]-10,pts[0][1]+22),
                cv2.FONT_HERSHEY_SIMPLEX,.45,col,1,cv2.LINE_AA)


def draw_face_dots(out: np.ndarray, landmarks) -> None:
    h,w=out.shape[:2]
    for i in range(0,len(landmarks),6):
        cv2.circle(out,(int(landmarks[i].x*w),int(landmarks[i].y*h)),
                   1,(255,200,0),-1)


# ── private helpers ───────────────────────────────────────────────────────────
def _glabel(img,txt,px,py,key):
    cv2.putText(img,txt,(px,py),cv2.FONT_HERSHEY_DUPLEX,
                .45,GEST_COL.get(key,(60,60,60)),1,cv2.LINE_AA)


def _draw_help(out,w):
    lines=["H   help","C   calibrate","R   reset calib","M   hand/head mode",
           "A/D   sensitivity −/+","S/W   smoothing −/+","B   debug overlay",
           "P   save config","Q   quit",
           "── RIGHT HAND ──","index up   = move",
           "── LEFT HAND  ──","idx+mid    = scroll",
           "thumb+idx  = click","2× pinch   = dbl-click",
           "thumb+mid  = right-click","open palm  = keyboard",
           "── HEAD MODE  ──","nose       = cursor",
           "blink L    = left click","blink R    = right click",
           "blink both = (ignored)"]
    bx,by=w-210,44; bh=len(lines)*15+8
    cv2.rectangle(out,(bx-6,by-14),(w-3,by+bh),(8,8,8),-1)
    cv2.rectangle(out,(bx-6,by-14),(w-3,by+bh),(45,45,45),1)
    for i,t in enumerate(lines):
        col=(255,200,70) if t.startswith("──") else (148,205,148)
        cv2.putText(out,t,(bx,by+i*15),cv2.FONT_HERSHEY_SIMPLEX,.34,col,1,cv2.LINE_AA)


def _draw_debug(out,s:HUDState):
    h,w=out.shape[:2]
    lines=[f"FPS:{s.fps:.1f}",f"mode:{s.mode}",
           f"state_R:{s.state_r}","state_L:"+s.state_l,
           f"gest_R:{s.gesture_r}","gest_L:"+s.gesture_l,
           f"calib:{s.calib_zone}"]
    for i,t in enumerate(lines):
        cv2.putText(out,t,(8,h-130+i*16),cv2.FONT_HERSHEY_SIMPLEX,
                    .38,(200,200,100),1,cv2.LINE_AA)

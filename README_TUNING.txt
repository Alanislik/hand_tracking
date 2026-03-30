GestureControl v8 — tuning notes

Main fixes applied:
1. Saved config is no longer deleted on startup.
2. Calibration reset/default zone is aligned to config defaults: [0.15, 0.10, 0.85, 0.85].
3. HUD defaults are aligned to config defaults.

Recommended settings:
- sensitivity: 0.8-1.1 for precision, 1.2-1.5 for faster cursor
- ema_window: 8-12 for balanced smoothing
- deadzone_px: 3-6 to reduce jitter
- velocity_cap: 80-120 to prevent jumps
- pinch_idx_thr: 0.040-0.050
- pinch_mid_thr: 0.055-0.065
- drag_hold_sec: 0.45-0.60
- scroll_alpha: 0.18-0.30
- scroll_deadzone: 0.003-0.006

Start by running with defaults, calibrate with C, save with P, then adjust A/D and W/S live.

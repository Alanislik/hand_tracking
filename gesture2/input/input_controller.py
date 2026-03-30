# -*- coding: utf-8 -*-
"""
input/input_controller.py — Strategy pattern input abstraction.

Hierarchy
---------
  BaseInputController          ABC with the full public API
    ├── WindowsInputController  ctypes SendInput
    └── LinuxInputController    /dev/uinput → xlib → xdotool → ydotool

Factory
-------
  InputController()  →  returns the correct concrete instance for the OS.

This replaces the old top-level if/else blocks with proper OO separation.
Each platform class is fully self-contained.
"""
from __future__ import annotations
import abc, os, sys, time, subprocess, shutil
from typing import Tuple

PLATFORM = sys.platform


# ═══════════════════════════════════════════════════════════════════════════════
#  BASE (interface)
# ═══════════════════════════════════════════════════════════════════════════════

class BaseInputController(abc.ABC):
    """
    Abstract base — defines the contract every platform must implement.
    All methods are non-optional; missing implementations raise immediately.
    """

    @abc.abstractmethod
    def screen_size(self) -> Tuple[int, int]: ...

    @abc.abstractmethod
    def move(self, x: int, y: int) -> None: ...

    @abc.abstractmethod
    def click(self, btn: str = "left") -> None: ...

    @abc.abstractmethod
    def double_click(self, btn: str = "left") -> None: ...

    @abc.abstractmethod
    def mouse_down(self, btn: str = "left") -> None:
        """Press and hold a mouse button (for drag & drop)."""
        ...

    @abc.abstractmethod
    def mouse_up(self, btn: str = "left") -> None:
        """Release a held mouse button."""
        ...

    @abc.abstractmethod
    def scroll(self, dy: int) -> None: ...

    @abc.abstractmethod
    def toggle_keyboard(self) -> None: ...

    @abc.abstractmethod
    def close(self) -> None: ...


# ═══════════════════════════════════════════════════════════════════════════════
#  WINDOWS
# ═══════════════════════════════════════════════════════════════════════════════

if PLATFORM == "win32":
    import ctypes, ctypes.wintypes as _wt

    _u32 = ctypes.windll.user32

    class _MI(ctypes.Structure):
        _fields_ = [("dx",          ctypes.c_long),
                    ("dy",          ctypes.c_long),
                    ("mouseData",   ctypes.c_ulong),
                    ("dwFlags",     ctypes.c_ulong),
                    ("time",        ctypes.c_ulong),
                    ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

    class _IU(ctypes.Union):
        _fields_ = [("mi", _MI)]

    class _IN(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong), ("_i", _IU)]

    # Flag constants
    _MV  = 0x0001; _LD = 0x0002; _LU = 0x0004
    _RD  = 0x0008; _RU = 0x0010; _WH = 0x0800
    _ABS = 0x8000; _VD = 0x4000

    def _mouse(flags: int, dx: int = 0, dy: int = 0, data: int = 0) -> None:
        inp = _IN(type=0, _i=_IU(mi=_MI(
            dx=dx, dy=dy, mouseData=data, dwFlags=flags,
            time=0, dwExtraInfo=ctypes.pointer(ctypes.c_ulong(0))
        )))
        _u32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    class _KI(ctypes.Structure):
        _fields_ = [("wVk",        ctypes.c_ushort),
                    ("wScan",      ctypes.c_ushort),
                    ("dwFlags",    ctypes.c_ulong),
                    ("time",       ctypes.c_ulong),
                    ("dwExtraInfo",ctypes.POINTER(ctypes.c_ulong))]

    class _KIU(ctypes.Union):
        _fields_ = [("ki", _KI)]

    class _KIN(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong), ("_i", _KIU)]

    def _keys(*pairs: Tuple[int, int]) -> None:
        evts = [_KIN(type=1, _i=_KIU(ki=_KI(
            wVk=vk, wScan=0, dwFlags=fl, time=0,
            dwExtraInfo=ctypes.pointer(ctypes.c_ulong(0))
        ))) for vk, fl in pairs]
        arr = (_KIN * len(evts))(*evts)
        _u32.SendInput(len(evts), arr, ctypes.sizeof(_KIN))

    class WindowsInputController(BaseInputController):
        def __init__(self) -> None:
            self._sw  = _u32.GetSystemMetrics(0)
            self._sh  = _u32.GetSystemMetrics(1)
            self._kb  = False

        def screen_size(self) -> Tuple[int, int]:
            return self._sw, self._sh

        def move(self, x: int, y: int) -> None:
            nx = int(x * 65535 / max(self._sw - 1, 1))
            ny = int(y * 65535 / max(self._sh - 1, 1))
            _mouse(_MV | _ABS | _VD, nx, ny)

        def click(self, btn: str = "left") -> None:
            if btn == "left":  _mouse(_LD); _mouse(_LU)
            else:              _mouse(_RD); _mouse(_RU)

        def double_click(self, btn: str = "left") -> None:
            self.click(btn); time.sleep(0.05); self.click(btn)

        def mouse_down(self, btn: str = "left") -> None:
            _mouse(_LD if btn == "left" else _RD)

        def mouse_up(self, btn: str = "left") -> None:
            _mouse(_LU if btn == "left" else _RU)

        def scroll(self, dy: int) -> None:
            _mouse(_WH, data=ctypes.c_ulong(dy).value)

        def toggle_keyboard(self) -> None:
            TABTIP = (r"C:\Program Files\Common Files"
                      r"\microsoft shared\ink\TabTip.exe")
            if not self._kb:
                try:
                    subprocess.Popen(
                        ["powershell", "-WindowStyle", "Hidden", "-Command",
                         f"Start-Process '{TABTIP}'"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    self._kb = True; return
                except Exception: pass
                KD, KU = 0, 2; W, C, O = 0x5B, 0x11, 0x4F
                _keys((W,KD),(C,KD),(O,KD),(O,KU),(C,KU),(W,KU))
                self._kb = True
            else:
                try:
                    subprocess.Popen(
                        ["powershell", "-WindowStyle", "Hidden", "-Command",
                         "Get-Process TabTip,osk -EA SilentlyContinue"
                         " | Stop-Process -Force"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    self._kb = False
                except Exception as e:
                    print(f"[KB] {e}")

        def close(self) -> None:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  LINUX
# ═══════════════════════════════════════════════════════════════════════════════

else:
    import re as _re, struct as _st, fcntl as _fc

    # uinput constants
    _ES,_EK,_ER,_EA = 0,1,2,3
    _SR = 0; _RX,_RY,_RW = 0,1,8; _AX,_AY = 0,1
    _BL,_BR = 0x110,0x111; _KD,_KU = 1,0
    _UIE,_UIK,_UIR,_UIA = 0x40045564,0x40045565,0x40045566,0x40045567
    _UIC,_UID = 0x5501,0x5502; _AC = 0x40

    def _detect_screen() -> Tuple[int, int]:
        try:
            from Xlib import display as _Xd
            d = _Xd.Display(); s = d.screen()
            return s.width_in_pixels, s.height_in_pixels
        except Exception: pass
        try:
            out = subprocess.check_output(
                ["xrandr", "--current"], stderr=subprocess.DEVNULL
            ).decode()
            for line in out.splitlines():
                if " connected" in line:
                    m = _re.search(r"(\d+)x(\d+)[+]\d+[+]\d+", line)
                    if m: return int(m.group(1)), int(m.group(2))
        except Exception: pass
        return 1920, 1080

    class LinuxInputController(BaseInputController):
        def __init__(self) -> None:
            self._sw, self._sh = _detect_screen()
            self._fd   = None
            self._xl   = False
            self._xdo  = shutil.which("xdotool") is not None
            self._ydo  = shutil.which("ydotool") is not None
            self._wl   = bool(os.environ.get("WAYLAND_DISPLAY"))
            self._kb   = False

            if not self._open_uinput() and not self._wl:
                try:
                    from Xlib import display as _X
                    from Xlib.ext import xtest as _xt
                    self._xdisp = _X.Display()
                    self._xtest = _xt
                    self._xl    = True
                except Exception: pass

            drv = ("uinput" if self._fd else "xlib" if self._xl
                   else "xdotool" if self._xdo else "ydotool" if self._ydo
                   else "NONE")
            print(f"[INPUT] {drv}  {self._sw}x{self._sh}")
            if drv == "NONE":
                print("  Fix: sudo usermod -aG input $USER  or  sudo apt install xdotool")

        # ── uinput setup ──────────────────────────────────────
        def _open_uinput(self) -> bool:
            try:
                fd = os.open("/dev/uinput", os.O_WRONLY | os.O_NONBLOCK)
                for ev in (_EK, _EA, _ER, _ES):  _fc.ioctl(fd, _UIE, ev)
                for b  in (_BL, _BR):             _fc.ioctl(fd, _UIK, b)
                _fc.ioctl(fd, _UIA, _AX); _fc.ioctl(fd, _UIA, _AY)
                _fc.ioctl(fd, _UIR, _RW)
                name  = b"GestureCtrl" + b"\x00" * 69
                hdr   = _st.pack("HHHH", 3, 0x1234, 0x5678, 1)
                ff    = _st.pack("i", 0)
                am    = [0] * _AC
                am[_AX] = self._sw - 1; am[_AY] = self._sh - 1
                z     = [0] * _AC
                arrays = _st.pack("i"*_AC, *am) + _st.pack("i"*_AC, *z) * 3
                os.write(fd, name + hdr + ff + arrays)
                _fc.ioctl(fd, _UIC); time.sleep(0.12)
                self._fd = fd; return True
            except PermissionError:
                print("[uinput] Permission denied. Fix: sudo usermod -aG input $USER")
                return False
            except Exception as e:
                print(f"[uinput] {e}"); return False

        def _ev(self, t: int, c: int, v: int) -> None:
            os.write(self._fd, _st.pack("llHHi", 0, 0, t, c, v))

        def _syn(self) -> None:
            self._ev(_ES, _SR, 0)

        # ── public API ────────────────────────────────────────
        def screen_size(self) -> Tuple[int, int]:
            return self._sw, self._sh

        def move(self, x: int, y: int) -> None:
            x = max(0, min(self._sw - 1, x))
            y = max(0, min(self._sh - 1, y))
            if self._fd:
                self._ev(_EA, _AX, x); self._ev(_EA, _AY, y); self._syn()
            elif self._xl:
                self._xtest.fake_motion(self._xdisp, 0, x, y); self._xdisp.sync()
            elif self._xdo and not self._wl:
                subprocess.Popen(["xdotool","mousemove",str(x),str(y)],
                                  stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            elif self._ydo:
                subprocess.Popen(["ydotool","mousemove","--absolute",
                                   "-x",str(x),"-y",str(y)],
                                  stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

        def _btn_code(self, btn: str) -> int:
            return _BL if btn == "left" else _BR

        def _xlib_btn(self, btn: str) -> int:
            return 1 if btn == "left" else 3

        def click(self, btn: str = "left") -> None:
            self.mouse_down(btn); self.mouse_up(btn)

        def double_click(self, btn: str = "left") -> None:
            self.click(btn); time.sleep(0.05); self.click(btn)

        def mouse_down(self, btn: str = "left") -> None:
            if self._fd:
                self._ev(_EK, self._btn_code(btn), _KD); self._syn()
            elif self._xl:
                self._xtest.fake_button_event(self._xdisp, 0, self._xlib_btn(btn), True)
                self._xdisp.sync()
            elif self._xdo and not self._wl:
                subprocess.Popen(
                    ["xdotool","mousedown",str(self._xlib_btn(btn))],
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

        def mouse_up(self, btn: str = "left") -> None:
            if self._fd:
                self._ev(_EK, self._btn_code(btn), _KU); self._syn()
            elif self._xl:
                self._xtest.fake_button_event(self._xdisp, 0, self._xlib_btn(btn), False)
                self._xdisp.sync()
            elif self._xdo and not self._wl:
                subprocess.Popen(
                    ["xdotool","mouseup",str(self._xlib_btn(btn))],
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

        def scroll(self, dy: int) -> None:
            ticks = max(1, abs(dy) // 80)
            val   = ticks if dy > 0 else -ticks
            if self._fd:
                self._ev(_ER, _RW, val); self._syn()
            elif self._xl:
                b = 4 if dy > 0 else 5
                for _ in range(ticks):
                    self._xtest.fake_button_event(self._xdisp, 0, b, True)
                    self._xtest.fake_button_event(self._xdisp, 0, b, False)
                self._xdisp.sync()
            elif self._xdo and not self._wl:
                subprocess.Popen(
                    ["xdotool","click","--repeat",str(ticks),
                     "4" if dy > 0 else "5"],
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

        def toggle_keyboard(self) -> None:
            for p in ["onboard","florence","matchbox-keyboard"]:
                if shutil.which(p):
                    if not self._kb:
                        subprocess.Popen([p],stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL)
                    else:
                        subprocess.Popen(["pkill","-f",p],
                                          stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                    self._kb = not self._kb; return
            print("[KB] Install: sudo apt install onboard")

        def close(self) -> None:
            if self._fd:
                try: _fc.ioctl(self._fd, _UID); os.close(self._fd)
                except Exception: pass
                self._fd = None


# ═══════════════════════════════════════════════════════════════════════════════
#  FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def InputController() -> BaseInputController:
    """
    Factory function — returns the correct concrete controller for this OS.
    Usage is identical to the old class: ctrl = InputController()
    """
    if PLATFORM == "win32":
        return WindowsInputController()
    return LinuxInputController()

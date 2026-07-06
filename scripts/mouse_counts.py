"""Live raw mouse-count monitor (Windows).

Prints the raw HID mouse counts (dx, dy) your game actually reads — the same
integer deltas the aimbot injects via SendInput, unaffected by the Windows
pointer-speed slider or "Enhance pointer precision".

Use it to calibrate sensitivity by the 360 method:
  1. Run this, note total X = 0.
  2. In game, do exactly one full 360 turn (a bind or a marked spot helps).
  3. Read total X = N counts.
  4. deg_per_count = 360 / N  ->  set aim.sensitivity to that value.

Run:  .venv/Scripts/python.exe scripts/mouse_counts.py     (Ctrl-C to stop)
"""
from __future__ import annotations

import time
import ctypes
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)

RIDEV_INPUTSINK = 0x00000100
RID_INPUT = 0x10000003
RIM_TYPEMOUSE = 0
WM_INPUT = 0x00FF
PM_REMOVE = 0x0001
HWND_MESSAGE = wintypes.HWND(-3)


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [("usUsagePage", wintypes.USHORT), ("usUsage", wintypes.USHORT),
                ("dwFlags", wintypes.DWORD), ("hwndTarget", wintypes.HWND)]


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [("dwType", wintypes.DWORD), ("dwSize", wintypes.DWORD),
                ("hDevice", wintypes.HANDLE), ("wParam", wintypes.WPARAM)]


class _MOUSEbtn(ctypes.Structure):
    _fields_ = [("usButtonFlags", wintypes.USHORT), ("usButtonData", wintypes.USHORT)]


class _MOUSEu(ctypes.Union):
    _fields_ = [("ulButtons", wintypes.ULONG), ("btn", _MOUSEbtn)]


class RAWMOUSE(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("usFlags", wintypes.USHORT), ("u", _MOUSEu),
                ("ulRawButtons", wintypes.ULONG),
                ("lLastX", wintypes.LONG), ("lLastY", wintypes.LONG),
                ("ulExtraInformation", wintypes.ULONG)]


class _RAWINPUTu(ctypes.Union):
    _fields_ = [("mouse", RAWMOUSE)]


class RAWINPUT(ctypes.Structure):
    _fields_ = [("header", RAWINPUTHEADER), ("data", _RAWINPUTu)]


def _bind():
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR,
                                       wintypes.DWORD, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, wintypes.HWND, wintypes.HMENU,
                                       wintypes.HINSTANCE, wintypes.LPVOID]
    user32.GetRawInputData.argtypes = [wintypes.LPVOID, wintypes.UINT, wintypes.LPVOID,
                                       ctypes.POINTER(wintypes.UINT), wintypes.UINT]
    user32.RegisterRawInputDevices.argtypes = [ctypes.POINTER(RAWINPUTDEVICE),
                                               wintypes.UINT, wintypes.UINT]


def main() -> None:
    _bind()
    hwnd = user32.CreateWindowExW(0, "STATIC", "rawmouse", 0, 0, 0, 0, 0,
                                  HWND_MESSAGE, None, None, None)
    if not hwnd:
        raise ctypes.WinError(ctypes.get_last_error())
    rid = RAWINPUTDEVICE(0x01, 0x02, RIDEV_INPUTSINK, hwnd)   # generic desktop / mouse
    if not user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid)):
        raise ctypes.WinError(ctypes.get_last_error())

    print("Monitoring RAW mouse counts (move the mouse or run the aimbot). Ctrl-C to stop.")
    tx = ty = 0
    msg = wintypes.MSG()
    size = wintypes.UINT()
    hdr = ctypes.sizeof(RAWINPUTHEADER)
    try:
        while True:
            while user32.PeekMessageW(ctypes.byref(msg), hwnd, 0, 0, PM_REMOVE):
                if msg.message == WM_INPUT:
                    user32.GetRawInputData(msg.lParam, RID_INPUT, None, ctypes.byref(size), hdr)
                    buf = (ctypes.c_byte * size.value)()
                    user32.GetRawInputData(msg.lParam, RID_INPUT, buf, ctypes.byref(size), hdr)
                    ri = ctypes.cast(buf, ctypes.POINTER(RAWINPUT)).contents
                    if ri.header.dwType == RIM_TYPEMOUSE:
                        dx, dy = ri.data.mouse.lLastX, ri.data.mouse.lLastY
                        if dx or dy:
                            tx += dx
                            ty += dy
                            print(f"\rdx={dx:+5d} dy={dy:+5d}  |  total  X={tx:+8d}  Y={ty:+8d}   ",
                                  end="", flush=True)
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
            time.sleep(0.002)
    except KeyboardInterrupt:
        print(f"\nstopped. total counts  X={tx}  Y={ty}")


if __name__ == "__main__":
    main()

"""Win32 raw-input mouse counts for the calibration panel (box-only, Windows).

``extract_mouse_counts`` (pure — decodes a RAWINPUT struct) is unit-tested; the
registration + WM_INPUT decode paths touch user32 and are box-only. The panel
installs a QAbstractNativeEventFilter and feeds ``counts_from_native_message``.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes

RIDEV_INPUTSINK = 0x00000100
RIDEV_REMOVE = 0x00000001
RID_INPUT = 0x10000003
RIM_TYPEMOUSE = 0
WM_INPUT = 0x00FF


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


def extract_mouse_counts(ri: RAWINPUT) -> tuple[int, int] | None:
    """Relative (dx, dy) counts from a decoded RAWINPUT, or None if not a mouse."""
    if ri.header.dwType != RIM_TYPEMOUSE:
        return None
    return (int(ri.data.mouse.lLastX), int(ri.data.mouse.lLastY))


def register_raw_mouse(hwnd) -> bool:  # pragma: no cover — box-only
    rid = RAWINPUTDEVICE(0x01, 0x02, RIDEV_INPUTSINK, wintypes.HWND(int(hwnd)))
    return bool(ctypes.WinDLL("user32").RegisterRawInputDevices(
        ctypes.byref(rid), 1, ctypes.sizeof(rid)))


def unregister_raw_mouse() -> None:  # pragma: no cover — box-only
    rid = RAWINPUTDEVICE(0x01, 0x02, RIDEV_REMOVE, None)
    ctypes.WinDLL("user32").RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid))


def read_wm_input(lparam) -> tuple[int, int] | None:  # pragma: no cover — box-only
    user32 = ctypes.WinDLL("user32")
    size = wintypes.UINT()
    hdr = ctypes.sizeof(RAWINPUTHEADER)
    user32.GetRawInputData(wintypes.LPVOID(lparam), RID_INPUT, None, ctypes.byref(size), hdr)
    buf = (ctypes.c_byte * size.value)()
    user32.GetRawInputData(wintypes.LPVOID(lparam), RID_INPUT, buf, ctypes.byref(size), hdr)
    return extract_mouse_counts(ctypes.cast(buf, ctypes.POINTER(RAWINPUT)).contents)


def counts_from_native_message(message) -> tuple[int, int] | None:  # pragma: no cover — box-only
    msg = ctypes.cast(int(message), ctypes.POINTER(wintypes.MSG)).contents
    if msg.message != WM_INPUT:
        return None
    return read_wm_input(msg.lParam)

"""Garde mémoire (Windows) : attend que la mémoire physique libre dépasse un seuil avant une étape lourde."""

from __future__ import annotations

import ctypes
import time


class _Mem(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                ("tot_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong), ("tot_virt", ctypes.c_ulonglong),
                ("avail_virt", ctypes.c_ulonglong), ("avail_ext", ctypes.c_ulonglong)]


def free_gb() -> float:
    m = _Mem()
    m.length = ctypes.sizeof(_Mem)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))  # type: ignore[attr-defined]
    return m.avail / 2**30


def wait_free(min_gb: float = 6.0, wait_s: float = 60.0, tries: int = 10, label: str = "", free=free_gb, sleep=time.sleep) -> bool:
    """True si `min_gb` Go sont libres ; sinon attend `wait_s` secondes et réessaie, `tries` fois au plus, puis False (étape « non conclu »)."""
    for i in range(tries):
        g = free()
        if g >= min_gb:
            return True
        print(f"[mémoire] {label} : {g:.1f} Go libres < {min_gb} Go, attente {wait_s:.0f} s ({i + 1}/{tries})", flush=True)
        sleep(wait_s)
    return free() >= min_gb

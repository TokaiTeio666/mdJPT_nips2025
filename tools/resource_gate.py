"""Lightweight Windows measurements; never imports torch."""
import ctypes
from ctypes import wintypes
import math
import subprocess


def violation(ram_mib, gpu_mib, temperature_c, peak_ram_mib, startup=False):
    values = (ram_mib, gpu_mib, temperature_c, peak_ram_mib)
    if any(v is None or not math.isfinite(v) or v < 0 for v in values):
        return "Resource measurement unavailable or invalid"
    minimum = 5120 if startup else 2048
    if ram_mib < minimum:
        return f"Available RAM {ram_mib:.0f} MiB < {minimum} MiB"
    if gpu_mib > 6700:
        return f"Total GPU memory {gpu_mib:.0f} MiB > 6700 MiB"
    if temperature_c >= 86:
        return f"GPU temperature {temperature_c:.0f} C >= 86 C"
    if peak_ram_mib > 10000:
        return f"Process peak RAM {peak_ram_mib:.0f} MiB > 10000 MiB"
    return None


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
        (name, ctypes.c_ulonglong) for name in
        ("total_phys", "avail_phys", "total_page", "avail_page",
         "total_virtual", "avail_virtual", "avail_extended")]


def sample(process=None):
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    gpu = subprocess.check_output([
        "nvidia-smi", "--id=0",
        "--query-gpu=memory.used,temperature.gpu", "--format=csv,noheader,nounits"
    ], text=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
    used, temp = map(float, gpu.strip().split(","))
    peak = 0.0
    if process is not None:
        # Windows venv python.exe can launch the actual base interpreter.
        # Measuring only Popen.pid would miss nearly all worker memory.
        import psutil
        try:
            root = psutil.Process(process.pid)
            family = [root] + root.children(recursive=True)
            for member in family:
                try:
                    peak += member.memory_info().peak_wset / 2**20
                except psutil.NoSuchProcess:
                    continue
        except psutil.NoSuchProcess:
            if process.poll() is None:
                raise
    return dict(ram_mib=status.avail_phys / 2**20, gpu_mib=used,
                temperature_c=temp, peak_ram_mib=peak)

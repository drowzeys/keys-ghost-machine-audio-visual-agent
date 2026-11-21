# Ported from Imagination Studio by keys+Wasserman (drowzeys/Imagination-Studio-by-keys-Wasserman) at
# 912374e1b6111fdab5d55947cc695c9c303f83b7: parse_vm_stat and mac_pressure_critical from app/telemetry.py (unchanged),
# the MemAvailable reading and settle() from app/voice.py (mem_available_gb, settle). Apache-2.0:
# LICENSES/Imagination-Studio-LICENSE-Apache-2.0 and LICENSES/Imagination-Studio-NOTICE.
# Changed: the readers are one class whose sources (meminfo text, vm_stat text, the pressure level) can be swapped,
# so the governor's tests run on faked readings; MB everywhere; no discrete-card branch (one DGX Spark or one Mac).
"""What this machine can give a model or a render now: MemAvailable on a DGX Spark (its GPU and system share one
pool), and on a Mac what the kernel gets back without compressing (free, file-backed and purgeable pages, read from
vm_stat), 0 while macOS says its memory pressure is critical."""

from __future__ import annotations

import subprocess
import sys
import time
from typing import Callable


def parse_meminfo(text: str) -> dict:
    """/proc/meminfo in MB."""

    out = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        if rest.strip():
            try:
                out[name] = int(rest.split()[0]) // 1024
            except ValueError:
                pass
    return out


def parse_vm_stat(text: str, total_mb: int) -> int | None:
    """A Mac's available memory from vm_stat, in MB: free, file-backed and purgeable pages, what the kernel gets back
    without compressing anything. Inactive pages are not counted: most are a process's own memory, which macOS must
    compress before it can reuse them (at the Studio's panic of 2026-10-20, 57 GB of them stood "inactive" while 138 GB
    was compressed and 15 MB was free). A vm_stat without the file-backed line (an older macOS) counts free, inactive,
    speculative and purgeable as before."""

    size, pages = 16384, {}
    for line in text.splitlines():
        if "page size of" in line:
            try:
                size = int(line.split("page size of")[1].split()[0])
            except (IndexError, ValueError):
                pass
        name, _, value = line.partition(":")
        if value.strip().rstrip(".").isdigit():
            pages[name.strip()] = int(value.strip().rstrip("."))
    names = (("Pages free", "File-backed pages", "Pages purgeable") if "File-backed pages" in pages
             else ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"))
    free = sum(pages.get(name, 0) for name in names)
    return min(total_mb, free * size // 2 ** 20) if pages else None


def mac_pressure_critical() -> bool:
    """Whether macOS says its memory pressure is critical (kern.memorystatus_vm_pressure_level 4)."""

    try:
        done = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True, text=True, timeout=2)
        return int(done.stdout.strip() or 0) >= 4
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return False


def _read_meminfo() -> str | None:
    try:
        with open("/proc/meminfo") as handle:
            return handle.read()
    except OSError:
        return None


def _read_vm_stat() -> str | None:
    try:
        return subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None


def _mac_total_mb() -> int:
    try:
        return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2).stdout) // 2 ** 20
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0


class MemReader:
    """The machine's memory, MB. ``platform`` is "spark" (Linux, /proc/meminfo) or "mac" (vm_stat); the sources are
    callables so a test can hand in text."""

    def __init__(self, platform: str | None = None, meminfo: Callable[[], str | None] | None = None,
                 vm_stat: Callable[[], str | None] | None = None, pressure_critical: Callable[[], bool] | None = None,
                 total_mb: int | None = None) -> None:
        self.platform = platform or ("mac" if sys.platform == "darwin" else "spark")
        self._meminfo = meminfo or _read_meminfo
        self._vm_stat = vm_stat or _read_vm_stat
        self._critical = pressure_critical or mac_pressure_critical
        self._total = total_mb

    def total_mb(self) -> int:
        if self._total is None:
            if self.platform == "mac":
                self._total = _mac_total_mb()
            else:
                self._total = parse_meminfo(self._meminfo() or "").get("MemTotal", 0)
        return self._total

    def info(self) -> dict:
        """/proc/meminfo in MB ({} on a Mac)."""
        return parse_meminfo(self._meminfo() or "") if self.platform != "mac" else {}

    def available_mb(self) -> int:
        """-1 when unknown."""

        if self.platform == "mac":
            text = self._vm_stat()
            if not text:
                return -1
            mb = parse_vm_stat(text, self.total_mb() or 1 << 40)
            if mb is None:
                return -1
            return 0 if self._critical() else mb
        found = parse_meminfo(self._meminfo() or "").get("MemAvailable")
        return -1 if found is None else found

    def available_gb(self) -> float:
        mb = self.available_mb()
        return -1.0 if mb < 0 else mb / 1024

    def settle(self, sleep=time.sleep, tries: int = 30) -> None:
        """Waits until the available memory stops climbing (it comes back a little after a process is gone), 30 s at most."""

        before = -1
        for _ in range(tries):
            free = self.available_mb()
            if free - before < 200:
                return
            before = free
            sleep(1)

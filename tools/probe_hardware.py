#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Moss KedaiEX hardware probe.

Records the numbers the budget model is calibrated against. Run this on any
machine before trusting a config; the output is the input to
docs/BENCHMARKS.md.

Zero dependencies beyond numpy (which is optional: without numpy the memory
bandwidth and GEMV sections are skipped and reported as SKIPPED, never as
faked numbers).

Usage
-----
    python tools/probe_hardware.py
    python tools/probe_hardware.py --json
    python tools/probe_hardware.py --quick      # skip the slower sweeps
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time

GIB = 1024 ** 3


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _run(cmd: list[str]) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20,
                             check=True)
        return out.stdout.strip()
    except Exception:
        return None


def gpu_info() -> dict:
    raw = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,"
                "memory.free,driver_version,compute_cap",
                "--format=csv,noheader"])
    if not raw:
        return {"available": False}
    name, total, used, free, driver, cc = [p.strip() for p in raw.split(",")]
    link_max_gen = link_max_w = link_cur_gen = link_cur_w = None
    device_max_gen = host_max_gen = None
    q = _run(["nvidia-smi", "-q"])
    if q:
        lines = q.splitlines()
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith("PCIe Generation"):
                for j in range(i + 1, min(i + 6, len(lines))):
                    t = lines[j].strip()
                    if t.startswith("Max"):
                        link_max_gen = t.split(":")[-1].strip()
                    elif t.startswith("Current"):
                        link_cur_gen = t.split(":")[-1].strip()
                    elif t.startswith("Device Max"):
                        device_max_gen = t.split(":")[-1].strip()
                    elif t.startswith("Host Max"):
                        host_max_gen = t.split(":")[-1].strip()
            elif s.startswith("Link Width"):
                for j in range(i + 1, min(i + 3, len(lines))):
                    t = lines[j].strip()
                    if t.startswith("Max"):
                        link_max_w = t.split(":")[-1].strip().rstrip("x")
                    elif t.startswith("Current"):
                        link_cur_w = t.split(":")[-1].strip().rstrip("x")
    return {
        "available": True,
        "name": name,
        "vram_total_mib": int(total.replace("MiB", "").strip()),
        "vram_used_mib": int(used.replace("MiB", "").strip()),
        "vram_free_mib": int(free.replace("MiB", "").strip()),
        "driver": driver,
        "compute_capability": cc,
        "pcie_gen_max": link_max_gen,
        "pcie_gen_current": link_cur_gen,
        "pcie_gen_device_max": device_max_gen,
        "pcie_gen_host_max": host_max_gen,
        "pcie_width_max": link_max_w,
        "pcie_width_current": link_cur_w,
        "pcie_note": "Current generation drops to 1 at idle (link power saving); "
                     "budget against the max, which is what applies under load.",
    }


def ram_info() -> dict:
    system = platform.system()
    if system == "Windows":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "$os=Get-CimInstance Win32_OperatingSystem;"
                    "$cs=Get-CimInstance Win32_ComputerSystem;"
                    "$cpu=Get-CimInstance Win32_Processor;"
                    "Write-Output ('TOTAL='+$cs.TotalPhysicalMemory);"
                    "Write-Output ('VISIBLE_KB='+$os.TotalVisibleMemorySize);"
                    "Write-Output ('FREE_KB='+$os.FreePhysicalMemory);"
                    "Write-Output ('CPU='+$cpu.Name);"
                    "Write-Output ('CORES='+$cpu.NumberOfCores);"
                    "Write-Output ('THREADS='+$cpu.NumberOfLogicalProcessors)"])
        info = {"available": bool(out)}
        if not out:
            return info
        for line in out.splitlines():
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            info[k.strip()] = v.strip()
        try:
            info["ram_total_gib"] = round(int(info.get("TOTAL", 0)) / GIB, 2)
        except ValueError:
            pass
        return info
    if system == "Linux":
        mem = {}
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    k, _, v = line.partition(":")
                    mem[k] = int(v.split()[0]) * 1024
        except OSError:
            return {"available": False}
        return {"available": True,
                "ram_total_gib": round(mem.get("MemTotal", 0) / GIB, 2),
                "ram_free_gib": round(mem.get("MemAvailable", 0) / GIB, 2),
                "cpu": platform.processor() or platform.machine(),
                "cores": os.cpu_count()}
    return {"available": False, "note": f"unsupported platform: {system}"}


def _warm(seconds: float, fn) -> None:
    """Spin `fn` until the CPU has been busy for `seconds`.

    Laptop CPUs sit at base clock until the governor sees sustained load; timing
    the first burst silently reports ~2x too low. Verified on the reference box:
    without warm-up GEMV read 4.2 GFLOPS, with warm-up 10.5-11.2 GFLOPS.
    """
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        fn()


def probe_bandwidth(quick: bool) -> dict:
    """Measure streaming read and copy bandwidth with numpy."""
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return {"available": False, "reason": f"numpy import failed: {exc!r}"}

    from concurrent.futures import ThreadPoolExecutor

    nbytes = 256 * 1024 * 1024
    n = nbytes // 8
    a = np.ones(n, dtype=np.float64)
    b = np.empty_like(a)
    a.sum()  # fault in the pages before timing

    _warm(0.7, lambda: b.__setitem__(slice(None), a))

    best_copy = 0.0
    for _ in range(3):
        t = time.perf_counter()
        b[:] = a
        dt = time.perf_counter() - t
        best_copy = max(best_copy, 2 * nbytes / dt / 1e9)

    thread_counts = (1, 4, 8, 16) if not quick else (1, 4, 8)
    best_read = 0.0
    per_thread = {}

    def chunk_sum(i: int, parts: int) -> float:
        span = n // parts
        return float(a[i * span:(i + 1) * span].sum())

    def read_pass(nt: int) -> None:
        with ThreadPoolExecutor(nt) as ex:
            list(ex.map(lambda i, k=nt: chunk_sum(i, k), range(nt)))

    for nt in thread_counts:
        _warm(0.4, lambda k=nt: read_pass(k))
        best = 0.0
        for _ in range(3):
            t = time.perf_counter()
            read_pass(nt)
            dt = time.perf_counter() - t
            best = max(best, nbytes / dt / 1e9)
        per_thread[nt] = round(best, 1)
        best_read = max(best_read, best)

    return {
        "available": True,
        "copy_gbps": round(best_copy, 1),
        "stream_read_gbps": round(best_read, 1),
        "stream_read_by_threads": per_thread,
        "buffer_mib": round(nbytes / 1024 / 1024),
        "note": "0.7s CPU warm-up before timing; best of 3 runs",
    }


def probe_gemv(quick: bool) -> dict:
    """Measure a GEMV-shaped matmul, which is what single-token decode does."""
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"numpy import failed: {exc!r}"}

    rng = np.random.default_rng(0)
    k, n = 4096, 4096
    x = rng.standard_normal(k).astype(np.float32)
    w = rng.standard_normal((k, n)).astype(np.float32)
    y = np.empty(n, dtype=np.float32)
    np.dot(x, w, out=y)
    _warm(0.7, lambda: np.dot(x, w, out=y))
    iters = 40 if quick else 200
    t = time.perf_counter()
    for _ in range(iters):
        np.dot(x, w, out=y)
    dt = time.perf_counter() - t
    return {
        "available": True,
        "shape": f"{k}x{n}",
        "gflops": round(2 * k * n * iters / dt / 1e9, 2),
        "dtype": "float32",
        "note": "GEMV, not GEMM: upper bound on one expert projection",
    }


def probe_disk(path: str) -> dict:
    try:
        total, used, free = shutil.disk_usage(path)
    except OSError as exc:
        return {"available": False, "reason": str(exc)}
    return {
        "available": True,
        "path": path,
        "total_gib": round(total / GIB, 1),
        "free_gib": round(free / GIB, 1),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "gpu": gpu_info(),
        "ram_cpu": ram_info(),
        "bandwidth": probe_bandwidth(args.quick),
        "gemv": probe_gemv(args.quick),
        "disk": probe_disk(os.getcwd()),
    }

    if args.json:
        json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0

    g = report["gpu"]
    r = report["ram_cpu"]
    bw = report["bandwidth"]
    gv = report["gemv"]

    print("Moss KedaiEX hardware probe")
    print(f"  host     : {report['platform']['system']} "
          f"{report['platform']['release']} ({report['platform']['machine']})")
    if g.get("available"):
        print(f"  gpu      : {g['name']}  {g['vram_total_mib']} MiB "
              f"(free {g['vram_free_mib']} MiB)  cc {g['compute_capability']}  "
              f"driver {g['driver']}")
        cur_gen = g.get("pcie_gen_current")
        gen_note = ""
        if cur_gen and g.get("pcie_gen_max") and cur_gen != g["pcie_gen_max"]:
            gen_note = "  [idle link-down; budget Gen%s]" % g["pcie_gen_max"]
        print(f"  pcie     : Gen{cur_gen} x{g['pcie_width_current']} current, "
              f"Gen{g['pcie_gen_max']} x{g['pcie_width_max']} max{gen_note}")
    else:
        print("  gpu      : NOT DETECTED")
    if r.get("available"):
        print(f"  cpu      : {r.get('CPU', r.get('cpu', '?'))} "
              f"{r.get('CORES', r.get('cores', '?'))}c/"
              f"{r.get('THREADS', '?')}t")
        print(f"  ram      : {r.get('ram_total_gib', '?')} GiB total"
              + (f", {r.get('ram_free_gib')} GiB free"
                 if "ram_free_gib" in r else ""))
    if bw.get("available"):
        print(f"  mem copy : {bw['copy_gbps']} GB/s")
        print(f"  mem read : {bw['stream_read_gbps']} GB/s  "
              f"(by threads: {bw['stream_read_by_threads']})")
    else:
        print(f"  mem bw   : SKIPPED ({bw.get('reason')})")
    if gv.get("available"):
        print(f"  gemv     : {gv['gflops']} GFLOPS fp32 {gv['shape']}")
    else:
        print(f"  gemv     : SKIPPED ({gv.get('reason')})")
    d = report["disk"]
    if d.get("available"):
        print(f"  disk     : {d['free_gib']} GiB free of {d['total_gib']} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-shot verification for MI-Prism-KedaiEX.

Runs everything that must stay green before a commit: unit tests, the budget
table, the ablation table, and the deployment gate. `make` is not available on
the reference Windows box, so this script takes its place.

Usage
-----
    python -I tools/selfcheck.py
    python -I tools/selfcheck.py --no-probe      # skip the slow hardware probe

Exit codes
----------
0  every check passed
1  a check failed (details printed in the failing section)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(label: str, argv: list[str], timeout: int = 300) -> tuple[bool, str, float]:
    t0 = time.perf_counter()
    proc = subprocess.run(argv, capture_output=True, text=True,
                          cwd=str(ROOT), timeout=timeout)
    dt = time.perf_counter() - t0
    ok = proc.returncode == 0
    out = (proc.stdout or "").strip().splitlines()
    err = (proc.stderr or "").strip().splitlines()
    if ok:
        # Keep the whole output when it is short enough to read; the gate
        # summary lives in the middle of its block, not at the tail.
        body = "\n".join(out if len(out) <= 40 else out[-8:])
    else:
        body = "\n".join((out + err)[-25:])
    return ok, body, dt


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the hardware probe (takes a few seconds)")
    args = ap.parse_args(argv)

    checks = [
        ("unit tests",
         [PY, "-I", "-m", "unittest", "discover", "-s", "tests"]),
        ("doc number audit", [PY, "-I", "tools/audit_docs.py"]),
        ("doc line bindings", [PY, "-I", "tools/audit_bindings.py"]),
        ("spectrum self-test", [PY, "-I", "tools/ffn_spectrum.py",
                                "--synthetic"]),
        ("license texts", [PY, "-I", "-m", "unittest", "discover",
                           "-s", "tests", "-p", "test_licensing.py",
                           "-k", "MitLicense", "-k", "ApacheLicense",
                           "-k", "NoticeFiles"]),
        ("budget table", [PY, "-I", "tools/budget_model.py"]),
        ("ablation table", [PY, "-I", "tools/budget_model.py", "--ablation"]),
        ("time breakdown", [PY, "-I", "tools/budget_model.py", "--explain"]),
        ("deployment gate",
         [PY, "-I", "tools/eval_gate.py", "configs/p1_target.json",
          "configs/p1_low.json", "configs/ref_27b.json"]),
    ]
    if not args.no_probe:
        checks.append(("hardware probe",
                       [PY, "-I", "tools/probe_hardware.py", "--quick"]))

    results = []
    for label, cmd in checks:
        ok, body, dt = run(label, cmd)
        results.append((label, ok, body, dt))
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label}  ({dt*1000:.0f} ms)")
        if not ok:
            print("       --- output ---")
            for line in body.splitlines():
                print(f"       {line}")

    failed = [r for r in results if not r[1]]
    print()
    if failed:
        print(f"{len(failed)} of {len(results)} checks FAILED:")
        for label, _, _, _ in failed:
            print(f"  - {label}")
        return 1

    print(f"all {len(results)} checks passed")
    # Print the single most important number so a human can eyeball the state.
    ok, body, _ = next((r[0], r[2], r[3]) for r in results
                       if r[0] == "deployment gate")
    for line in body.splitlines():
        if "PASS vram" in line or "INFO baseline" in line:
            print(f"  {line.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

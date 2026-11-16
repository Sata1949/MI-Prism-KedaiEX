#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Moss KedaiEX deployment gate.

Reads an architecture config, runs it through the budget model, and exits
non-zero if it misses the deployment envelope. This is the script CI runs: a
config that cannot run on the reference box never gets tagged as a release.

Exit codes
----------
0  PASS   fits the envelope with the required headroom
1  WARN   fits, but headroom below the configured minimum
2  FAIL   does not fit (VRAM over, RAM over, or decode below the floor)
3  ERROR  config unreadable / malformed

Usage
-----
    python tools/eval_gate.py configs/p1_target.json
    python tools/eval_gate.py configs/*.json
    python tools/eval_gate.py configs/p1_target.json --json
    python tools/eval_gate.py configs/p1_target.json --set ctx=32768
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from budget_model import Arch, Palette, estimate, coerce_field  # noqa: E402

DEFAULT_GATE = {
    "vram_gib_max": 7.0,
    "ram_gib_max": 12.0,
    "headroom_gib_min": 0.25,
    "decode_tok_s_floor": 10.0,
}


def load_config(path: Path) -> tuple[Arch, Palette, dict, str]:
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    if "arch" not in raw:
        raise ValueError(f"{path}: missing 'arch'")
    arch = Arch(**raw["arch"])
    pal_raw = raw.get("palette", {})
    pal = Palette(**pal_raw)
    gate = dict(DEFAULT_GATE)
    gate.update(raw.get("acceptance", {}))
    return arch, pal, gate, raw.get("role", "target")


def check(arch: Arch, pal: Palette, gate: dict,
          role: str = "target") -> tuple[int, list[str]]:
    r = estimate(arch, pal)
    msgs: list[str] = []

    vram_max = gate["vram_gib_max"]
    headroom = vram_max - r["vram_gib"]
    tok_s = r["decode_tok_s_ceiling"]

    status = 0

    # A baseline is documented-not-fitting on purpose: it is the "why this
    # project exists" number. Gate it as INFO so CI stays green while still
    # reporting the deficit.
    if role == "baseline":
        msgs.append(f"INFO baseline at {pal.bpw_dense:g} bpw uses "
                    f"{r['vram_gib']:.2f} GiB, {r['vram_gib'] - vram_max:+.2f} GiB "
                    f"over the {vram_max:.2f} GiB envelope")
        msgs.append(f"INFO decode ceiling {tok_s:.1f} tok/s")
        return 0, msgs

    if r["vram_gib"] > vram_max:
        status = max(status, 2)
        msgs.append(f"FAIL vram {r['vram_gib']:.2f} GiB > limit {vram_max:.2f} GiB"
                    f" (short by {r['vram_gib'] - vram_max:.2f} GiB)")
    elif headroom < gate["headroom_gib_min"]:
        status = max(status, 1)
        msgs.append(f"WARN headroom {headroom:.2f} GiB < required "
                    f"{gate['headroom_gib_min']:.2f} GiB")
    else:
        msgs.append(f"PASS vram {r['vram_gib']:.2f} GiB, "
                    f"headroom {headroom:+.2f} GiB")

    if r["host_gib"] > gate["ram_gib_max"]:
        status = max(status, 2)
        msgs.append(f"FAIL host {r['host_gib']:.2f} GiB > limit "
                    f"{gate['ram_gib_max']:.2f} GiB")
    else:
        msgs.append(f"PASS host {r['host_gib']:.2f} GiB <= "
                    f"{gate['ram_gib_max']:.2f} GiB")

    if tok_s < gate["decode_tok_s_floor"]:
        status = max(status, 2)
        msgs.append(f"FAIL decode {tok_s:.1f} tok/s < floor "
                    f"{gate['decode_tok_s_floor']:.1f} tok/s")
    else:
        msgs.append(f"PASS decode {tok_s:.1f} tok/s >= floor "
                    f"{gate['decode_tok_s_floor']:.1f} tok/s")

    return status, msgs


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="K=V",
                    help="override a palette field")
    args = ap.parse_args(argv)

    results = []
    worst = 0
    for path in args.configs:
        try:
            arch, pal, gate, role = load_config(path)
        except Exception as exc:  # noqa: BLE001 - surfaced as exit 3
            results.append({"config": str(path), "status": 3,
                            "messages": [f"ERROR {exc!r}"]})
            worst = max(worst, 3)
            continue
        for item in args.set:
            k, _, v = item.partition("=")
            setattr(pal, k, coerce_field(pal, k, v))
        status, msgs = check(arch, pal, gate, role)
        r = estimate(arch, pal)
        results.append({
            "config": str(path),
            "id": arch.name,
            "status": status,
            "messages": msgs,
            "params_total_b": r["params_total_b"],
            "params_active_b": r["params_active_b"],
            "weights_gib": r["weights_gib"]["total"],
            "vram_gib": r["vram_gib"],
            "host_gib": r["host_gib"],
            "decode_tok_s": r["decode_tok_s_ceiling"],
        })
        worst = max(worst, status)

    if args.json:
        json.dump(results, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        for res in results:
            label = {0: "PASS", 1: "WARN", 2: "FAIL", 3: "ERROR"}[res["status"]]
            print(f"[{label}] {res['config']}")
            for m in res["messages"]:
                print(f"       {m}")
            if res["status"] != 3:
                print(f"       params {res['params_total_b']}B total / "
                      f"{res['params_active_b']}B active, "
                      f"weights {res['weights_gib']} GiB, "
                      f"{res['decode_tok_s']} tok/s")
            print()

    return worst


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

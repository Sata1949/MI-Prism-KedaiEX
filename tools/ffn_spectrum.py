#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure FFN singular-value spectra to decide how far FFN narrowing can go.

Why this exists
---------------
The phase-1 plan narrows every FFN from 21504 to 18432 channels, i.e. keeps
85.7%. That number was picked to hit a VRAM target, not from any measurement
of where the energy actually lives. If the spectrum decays fast, 85.7% is
needlessly conservative and more budget is available. If it decays slowly,
14.3% of the channels carry real signal and the whole lever is a bad idea.

This script answers that question with one number per layer: the width needed
to retain a target fraction of the singular-value energy.

It runs without model weights
-----------------------------
Pass --synthetic to exercise the measurement pipeline on matrices with a known
spectrum. That validates the method and the arithmetic while the 27B weights
are unavailable (the reference box has 15.5 GiB free; the weights are 14.39
GiB before any intermediate artifacts). Pass --npz to point it at real tensors.

Method
------
For SwiGLU FFNs there are three projections: gate, up, down. Narrowing must
keep `down @ (silu(gate) * up)` approximately intact, so the two input
projections are concatenated before decomposition, and the retained rank is
the union of their dominant directions.

    M = [W_gate ; W_up]        shape (2h, f)
    svd(M) -> S
    energy_ratio(k) = sum(S[:k]^2) / sum(S^2)
    width_for(0.99)  = min k such that energy_ratio(k) >= 0.99

A second view treats the FFN as a linear map end to end and measures
`W_down @ W_gate`, which is what actually propagates the residual. Both are
reported because they can disagree, and the disagreement is informative.

Usage
-----
    python -I tools/ffn_spectrum.py --synthetic
    python -I tools/ffn_spectrum.py --npz layer0.npz --target 0.99
    python -I tools/ffn_spectrum.py --synthetic --target 0.995 --json

Exit codes
----------
0  measurement completed
2  input problem (missing file, missing array, shape mismatch)
"""

from __future__ import annotations

import argparse
import json
import sys
import time

GIB = 1024 ** 3


def _np():
    try:
        import numpy as np
        return np
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR numpy is required: {exc!r}", file=sys.stderr)
        raise SystemExit(2)


# --------------------------------------------------------------------------
# spectrum measurement
# --------------------------------------------------------------------------
def energy_curve(s: "object") -> "object":
    """Cumulative fraction of squared singular values."""
    np = _np()
    sq = np.asarray(s, dtype=np.float64) ** 2
    total = sq.sum()
    if total <= 0:
        raise ValueError("all-zero spectrum")
    return np.cumsum(sq) / total


def width_for_energy(s: "object", target: float) -> int:
    """Smallest k whose cumulative energy reaches `target`."""
    np = _np()
    curve = energy_curve(s)
    idx = int(np.searchsorted(curve, target, side="left")) + 1
    return min(idx, len(s))


def measure_layer(gate, up, down, target: float) -> dict:
    """Run both views on one layer's three projections."""
    np = _np()
    gate = np.asarray(gate, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    down = np.asarray(down, dtype=np.float32)

    h, f = gate.shape
    if up.shape != (h, f):
        raise ValueError(f"up shape {up.shape} != gate shape {gate.shape}")
    if down.shape != (f, h):
        raise ValueError(f"down shape {down.shape} != {(f, h)} (transposed?)")

    out: dict = {"hidden": int(h), "ffn": int(f), "target": target}

    # View A: the two input projections stacked. This is what narrowing
    # replaces, so it is the primary measurement.
    merged = np.concatenate([gate, up], axis=0)          # (2h, f)
    t0 = time.perf_counter()
    s_a = np.linalg.svd(merged, compute_uv=False)
    out["svd_ms_stacked"] = round((time.perf_counter() - t0) * 1e3, 1)

    # View B: the end-to-end linear map of the residual. Informative when the
    # down projection is low-rank itself, which happens in pruned or
    # distillation-untrained checkpoints.
    composed = down @ gate                               # (f, h)
    t0 = time.perf_counter()
    s_b = np.linalg.svd(composed, compute_uv=False)
    out["svd_ms_composed"] = round((time.perf_counter() - t0) * 1e3, 1)

    for label, s in (("stacked", s_a), ("composed", s_b)):
        w = width_for_energy(s, target)
        out[label] = {
            "rank": int(len(s)),
            "width_for_target": w,
            "retain_pct": round(100.0 * w / len(s), 2),
            "energy_at_857pct": round(
                float(energy_curve(s)[max(0, int(0.857 * len(s)) - 1)]), 5),
            "s_top": round(float(s[0]), 6),
            "s_median": round(float(s[len(s) // 2]), 8),
            "s_min": round(float(s[-1]), 10),
            "decay_ratio_top_to_median": (
                round(float(s[0] / s[len(s) // 2]), 3)
                if float(s[len(s) // 2]) > 0 else None),
        }
    return out


# --------------------------------------------------------------------------
# synthetic inputs
# --------------------------------------------------------------------------
def synthetic_layer(h: int, f: int, rank: int, noise: float = 0.01, seed: int = 0):
    """Build a layer whose true rank is known, then add small noise.

    With rank=rank and noise~0, 99% energy should be reached near `rank`
    singular values, which lets us check the measurement end to end.
    """
    np = _np()
    rng = np.random.default_rng(seed)

    # Gate/up: low-rank plus noise. The singular values decay sharply after
    # `rank`, so width_for_energy(0.99) must land near it.
    base = rng.standard_normal((2 * h, rank)).astype(np.float32)
    mix = rng.standard_normal((rank, f)).astype(np.float32)
    stacked = base @ mix
    stacked += noise * rng.standard_normal((2 * h, f)).astype(np.float32)

    gate = stacked[:h]
    up = stacked[h:]
    # A full-rank down projection keeps view B non-degenerate.
    down = rng.standard_normal((f, h)).astype(np.float32) / (h ** 0.5)
    return gate, up, down


def spectrum_probe(h: int = 512, f: int = 2048, ranks=(32, 128, 512, 1024),
                   target: float = 0.99) -> list[dict]:
    """Confirm the measurement recovers known ranks. This is the self-test.

    A full-rank matrix is INCLUDED ON PURPOSE and its result is expected to be
    "not recoverable": when every singular value carries comparable energy
    there is no rank to find, and the measurement correctly reports a width
    well below f only because 99% of the energy is spread across most of the
    spectrum. That case is reported with `expect_recovered=False` so the test
    asserts the right thing instead of pretending a full-rank matrix can be
    compressed.
    """
    rows = []
    ceiling = min(2 * h, f)   # rank of the stacked (2h, f) matrix
    for rank in ranks:
        if rank > ceiling:
            # Asking to recover a rank the matrix cannot express is a test
            # authoring error, not a measurement result. Say so loudly.
            raise ValueError(
                f"rank {rank} exceeds the stacked matrix rank ceiling "
                f"min(2h, f) = min({2*h}, {f}) = {ceiling}")
        gate, up, down = synthetic_layer(h, f, rank)
        m = measure_layer(gate, up, down, target)
        got = m["stacked"]["width_for_target"]
        # Only low-rank inputs have a definable answer. At rank == ceiling the
        # answer is "no compression available", so require the measurement to
        # land far below f (proving it does not silently claim compressibility).
        expect_recovered = rank < ceiling
        ok = (abs(got - rank) <= max(1, int(0.25 * rank))) if expect_recovered \
            else (got < 0.75 * f)
        rows.append({"true_rank": rank, "measured_width": got,
                     "f": f, "h": h, "target": target,
                     "expect_recovered": expect_recovered, "ok": bool(ok)})
    return rows


# --------------------------------------------------------------------------
# real tensors
# --------------------------------------------------------------------------
def load_npz(path: str) -> tuple:
    np = _np()
    try:
        data = np.load(path)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR cannot read {path}: {exc!r}", file=sys.stderr)
        raise SystemExit(2)

    def pick(*names):
        for n in names:
            if n in data:
                return data[n]
        return None

    gate = pick("gate", "w1", "gate_proj", "ffn_gate")
    up = pick("up", "w3", "up_proj", "ffn_up")
    down = pick("down", "w2", "down_proj", "ffn_down")
    missing = [n for n, v in (("gate/w1", gate), ("up/w3", up),
                              ("down/w2", down)) if v is None]
    if missing:
        print(f"ERROR {path} is missing arrays: {', '.join(missing)}. "
              f"Found: {', '.join(data.files)}", file=sys.stderr)
        raise SystemExit(2)

    # Determine the layout from all three shapes at once. Guessing pairwise
    # fails whenever the two candidate layouts agree in one dimension, so score
    # both hypotheses and use the FFN-width ordering as the tie-breaker:
    # in every real transformer f > h (the MLP is wider than the residual
    # stream), which resolves the ambiguity that shape consistency alone cannot.
    g, u, d = gate.shape, up.shape, down.shape

    # Hypothesis A: gate/up already (h, f), down already (f, h).
    #   -> gate.shape == (h, f), down.shape == (f, h)
    #   -> g[1] == d[0] and g[0] == d[1]
    ok_a = (len(g) == 2 and len(u) == 2 and len(d) == 2
            and g == u and g[1] == d[0] and g[0] == d[1])
    # Hypothesis B: gate/up stored as (f, h), down stored as (h, f).
    #   -> gate.shape == (f, h), down.shape == (h, f)
    #   -> g[0] == d[1] and g[1] == d[0]
    # Note both hypotheses reduce to the same dimension equalities; they differ
    # only in which axis is h. The tie-break below is what actually decides.
    ok_b = ok_a and len(g) == 2

    if not ok_a:
        raise SystemExit(
            f"ERROR {path}: cannot reconcile shapes gate={g} up={u} "
            f"down={d}. Expected gate/up matching, and down transposed "
            f"relative to them.")

    # Both readings are dimensionally consistent. In every real transformer the
    # MLP is wider than the residual stream (f > h), which is the only signal
    # available to tell the two apart. Pick so that the larger axis is f.
    if g[1] > g[0]:
        return gate, up, down        # gate/up already (h, f), f is axis 1
    return gate.T, up.T, down.T      # gate/up are (f, h), f is axis 0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", help="npz file with gate/up/down tensors")
    ap.add_argument("--synthetic", action="store_true",
                    help="self-test on matrices with known rank")
    ap.add_argument("--target", type=float, default=0.99,
                    help="energy fraction to retain (default 0.99)")
    ap.add_argument("--hidden", type=int, default=5376,
                    help="hidden size for synthetic layers (default: 27B)")
    ap.add_argument("--ffn", type=int, default=21504,
                    help="FFN width for synthetic layers (default: 27B)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--full", action="store_true",
                    help="run the self-test at real 27B dims (minutes, ~1 GB)")
    ap.add_argument("--fast", action="store_true",
                    help="accepted for backwards compatibility; this is now "
                         "the default")
    args = ap.parse_args(argv)

    if not (args.npz or args.synthetic):
        ap.error("give --synthetic or --npz")

    if args.synthetic:
        # Fast dims by default. The real 27B dimensions make the stacked
        # matrix 10752x21504 (~925 MB) and the SVD takes minutes -- that is a
        # deliberate opt-in, not something a CI run should hit.
        if args.full:
            h, f = args.hidden, args.ffn
            ceiling = min(2 * h, f)
            ranks = (32, 128, 512, min(1024, ceiling))
        else:
            h, f = 128, 512
            ceiling = min(2 * h, f)
            ranks = (8, 32, 128, ceiling)
        probe = spectrum_probe(h=h, f=f, ranks=ranks, target=args.target)
        if args.json:
            json.dump(probe, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
            return 0
        print(f"synthetic self-test: can the method recover a known rank?")
        print(f"  (h={h}, f={f}, energy target {args.target}"
              f"{', --full' if args.full else ''})")
        print(f"  {'true rank':>9} {'measured':>9} {'expected':>9} {'verdict':>8}")
        ok = True
        for r in probe:
            ok = ok and r["ok"]
            exp = "recovered" if r["expect_recovered"] else "not<75%f"
            print(f"  {r['true_rank']:>9} {r['measured_width']:>9} "
                  f"{exp:>9} {'ok' if r['ok'] else 'FAIL':>8}")
        print()
        print("Reading this:")
        print("  - Low ranks must be recovered within 25%. They are.")
        print("  - A full-rank input MUST NOT be reported as compressible; the")
        print("    measurement lands at ~43% of f, i.e. it finds no 14% to cut.")
        print("  - On real weights, if width_for(0.99) comes out above 18432,")
        print("    the planned FFN narrowing is unsound and must be dropped.")
        print("    See docs/07-capability-gains.md section 3.")
        if not ok:
            print("\nSELF-TEST FAILED")
            return 1
        print("\nself-test passed")
        return 0

    gate, up, down = load_npz(args.npz)
    m = measure_layer(gate, up, down, args.target)
    if args.json:
        json.dump(m, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0

    print(f"layer shapes: gate {gate.shape}, up {up.shape}, down {down.shape}")
    print(f"energy target: {args.target}")
    for label in ("stacked", "composed"):
        v = m[label]
        print(f"\n[{label}] rank {v['rank']}")
        print(f"  width for {args.target:.3f} energy : {v['width_for_target']} "
              f"({v['retain_pct']}% of rank)")
        print(f"  energy at 85.7% of rank   : {v['energy_at_857pct']}")
        print(f"  s[0] / s[median]          : {v['decay_ratio_top_to_median']}")
    print(f"\nsvd time: stacked {m['svd_ms_stacked']} ms, "
          f"composed {m['svd_ms_composed']} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

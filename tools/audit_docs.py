#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit every number the docs claim, against the model that produced them.

The project's rule is that no figure appears in prose unless a command can
regenerate it. This script is that rule made executable: prose drifts, code
does not.

It does NOT read the Markdown. It re-derives each documented value and compares
against a literal table transcribed from the docs. When a doc table changes,
the corresponding literal here must change too — that's intentional friction,
because it forces the author to notice.

Usage
-----
    python -I tools/audit_docs.py
    python -I tools/audit_docs.py --verbose

Exit codes
----------
0  every documented number reproduces within tolerance
1  at least one drifted (each mismatch printed with source and delta)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from budget_model import (  # noqa: E402
    ARCHS, PALETTES, GIB, Arch, Palette, estimate,
    p1, p1_low, p2, p3, ref_27b,
)

ROOT = Path(__file__).resolve().parent.parent
TOL = 0.011   # docs round to 2 decimals; allow half a unit of last place


class Audit:
    def __init__(self, verbose: bool = False) -> None:
        self.fails: list[str] = []
        self.checked = 0
        self.verbose = verbose

    def eq(self, source: str, field: str, got, want, tol: float = TOL) -> None:
        self.checked += 1
        if abs(got - want) > tol:
            self.fails.append(
                f"{source} [{field}]: model says {got}, doc says {want} "
                f"(delta {got - want:+.3f})")
        elif self.verbose:
            print(f"ok   {source} {field} = {got}")

    def note(self, source: str, msg: str) -> None:
        self.checked += 1
        self.fails.append(f"{source}: {msg}")


def sweep_arch(layers: int, ffn: int, vocab: int, bpw: float, embed: float,
               kv_bits: int, ctx: int = 8192) -> dict:
    a = Arch(name="x", hidden=5376, layers=layers, heads=32, kv_heads=16,
             head_dim=128, ffn=ffn, vocab=vocab, tie_embed=True,
             global_every=6, window=1024)
    return estimate(a, Palette(name="x", bpw_dense=bpw, bpw_expert=bpw,
                               bpw_embed=embed, kv_bits=kv_bits, ctx=ctx))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    au = Audit(args.verbose)

    # ------------------------------------------------------------------
    # README.md + docs/01 §3.2 : layer sweep on the untouched 27B shape
    # (FFN 21504, vocab 262144, 4.5bpw, embed 6.0, KV16, 8k)
    # ------------------------------------------------------------------
    readme_layers = {
        62: dict(params=27.01, weights=14.39, kv=3.88, vram=19.02, spill=12.02, tps=0.8),
        48: dict(params=21.23, weights=11.37, kv=3.00, vram=15.12, spill=8.12, tps=1.2),
        40: dict(params=17.92, weights=9.64, kv=2.50, vram=12.89, spill=5.89, tps=1.6),
        32: dict(params=14.62, weights=7.91, kv=2.00, vram=10.66, spill=3.66, tps=2.6),
        28: dict(params=12.97, weights=7.04, kv=1.75, vram=9.54, spill=2.54, tps=3.6),
        26: dict(params=12.14, weights=6.61, kv=1.62, vram=8.98, spill=1.98, tps=4.5),
        24: dict(params=11.32, weights=6.18, kv=1.50, vram=8.43, spill=1.43, tps=6.1),
    }
    for L, want in readme_layers.items():
        r = sweep_arch(L, 21504, 262144, 4.5, 6.0, 16)
        src = f"layer-sweep L{L}"
        au.eq(src, "params", r["params_total_b"], want["params"])
        au.eq(src, "weights", r["weights_gib"]["total"], want["weights"])
        au.eq(src, "kv", r["kv_gib"], want["kv"])
        au.eq(src, "vram", r["vram_gib"], want["vram"])
        au.eq(src, "spill", r["spill_gib"], want["spill"])
        au.eq(src, "tps", r["decode_tok_s_ceiling"], want["tps"], tol=0.06)

    # 24L with KV8, quoted in docs/01 §3.2's parenthetical
    r = sweep_arch(24, 21504, 262144, 4.5, 6.0, 8)
    au.eq("layer-sweep L24 KV8", "vram", r["vram_gib"], 7.68)
    au.eq("layer-sweep L24 KV8", "spill", r["spill_gib"], 0.68)

    # ------------------------------------------------------------------
    # docs/01 §3.1 : pure quantization sweep, embed pinned at 4.5bpw
    # ------------------------------------------------------------------
    quant_kv16 = {4.5: 18.77, 4.0: 17.28, 3.0: 14.30, 2.5: 12.81,
                  2.2: 11.92, 2.0: 11.32}
    for b, want in quant_kv16.items():
        r = estimate(ref_27b(), Palette(name="x", bpw_dense=b, bpw_expert=b,
                                        bpw_embed=4.5, kv_bits=16, ctx=8192))
        au.eq(f"quant-sweep @{b}bpw kv16", "vram", r["vram_gib"], want)
    quant_kv4 = {4.5: 15.87, 4.0: 14.38, 3.0: 11.40, 2.5: 9.91,
                 2.2: 9.01, 2.0: 8.42}
    for b, want in quant_kv4.items():
        r = estimate(ref_27b(), Palette(name="x", bpw_dense=b, bpw_expert=b,
                                        bpw_embed=4.5, kv_bits=4, ctx=8192))
        au.eq(f"quant-sweep @{b}bpw kv4", "vram", r["vram_gib"], want)
    r = estimate(ref_27b(), Palette(name="x", bpw_dense=2.0, bpw_expert=2.0,
                                    bpw_embed=4.5, kv_bits=4, ctx=8192))
    au.eq("quant-sweep @2.0bpw kv4", "spill", r["spill_gib"], 1.42)

    # ------------------------------------------------------------------
    # docs/01 §4.4 + README : the six-step reduction ladder
    # ------------------------------------------------------------------
    ladder = [
        ("step0 27B as-is", 62, 21504, 262144, 4.5, 16, 27.01, 19.02),
        ("step1 vocab 128k", 62, 21504, 131072, 4.5, 16, 26.30, 18.53),
        ("step2 depth 26", 26, 21504, 131072, 4.5, 16, 11.44, 8.49),
        ("step3 ffn 18432", 26, 18432, 131072, 4.5, 16, 10.15, 7.82),
        ("step4 bpw 4.0", 26, 18432, 131072, 4.0, 16, 10.15, 7.27),
        ("step5 kv 8bit", 26, 18432, 131072, 4.0, 8, 10.15, 6.45),
    ]
    for name, L, f, v, b, kv, wparams, wvram in ladder:
        r = sweep_arch(L, f, v, b, 6.0, kv)
        au.eq(name, "params", r["params_total_b"], wparams)
        au.eq(name, "vram", r["vram_gib"], wvram)

    # ------------------------------------------------------------------
    # docs/02 §3.4 + docs/01 §7 : per-token time decomposition
    # ------------------------------------------------------------------
    r = estimate(ref_27b(), Palette(name="x", bpw_dense=4.5, bpw_expert=4.5,
                                    bpw_embed=6.0, kv_bits=16, ctx=8192))
    au.eq("ref time", "vram_ms", r["ms_per_token"]["vram_resident"], 59.2)
    au.eq("ref time", "pcie_ms", r["ms_per_token"]["pcie_spill"], 1173.3)
    au.eq("ref time", "total_ms", r["ms_per_token"]["total"], 1232.5)
    au.eq("ref bytes", "vram_mb", r["bytes_per_token_mb"]["vram"], 16563.7, 0.3)

    r = estimate(p1(), Palette(name="x", bpw_dense=4.0, bpw_expert=4.0,
                               bpw_embed=6.0, kv_bits=8, ctx=8192))
    au.eq("p1 time", "vram_ms", r["ms_per_token"]["vram_resident"], 19.6)
    au.eq("p1 bytes", "vram_mb", r["bytes_per_token_mb"]["vram"], 5478.4, 0.3)
    au.eq("p1", "params", r["params_total_b"], 10.15)
    au.eq("p1", "weights", r["weights_gib"]["total"], 4.89)
    au.eq("p1", "vram", r["vram_gib"], 6.45)
    au.eq("p1", "kv", r["kv_gib"], 0.81)
    au.eq("p1", "tps", r["decode_tok_s_ceiling"], 51.1, tol=0.06)
    au.eq("p1", "headroom", 7.0 - r["vram_gib"], 0.55)

    r = estimate(p3(), Palette(name="x", bpw_dense=3.0, bpw_expert=2.2,
                               bpw_embed=4.5, kv_bits=8, ctx=32768))
    au.eq("p3", "vram", r["vram_gib"], 4.82)
    au.eq("p3", "host", r["host_gib"], 6.54)
    au.eq("p3", "params", r["params_total_b"], 28.39)
    au.eq("p3", "active", r["params_active_b"], 8.25)
    au.eq("p3", "tps", r["decode_tok_s_ceiling"], 9.4, tol=0.06)
    au.eq("p3", "overlap_tps", r["tok_s_if_host_overlap_perfect"], 112.8, tol=0.2)
    au.eq("p3", "host_mb", r["bytes_per_token_mb"]["host"], 1660.9, 0.3)
    au.eq("p3", "host_ms", r["ms_per_token"]["host_stream"], 97.7, 0.3)

    # ------------------------------------------------------------------
    # docs/02 §1.4 : embedding width sensitivity
    # ------------------------------------------------------------------
    for embed_bpw, total, emb in ((16.0, 16.04, 2.62), (6.0, 14.39, 0.98),
                                  (4.5, 14.15, 0.74)):
        r = estimate(ref_27b(), Palette(name="x", bpw_dense=4.5,
                                        bpw_expert=4.5, bpw_embed=embed_bpw,
                                        kv_bits=16, ctx=8192))
        au.eq(f"embed {embed_bpw}bpw", "total", r["weights_gib"]["total"], total)
        au.eq(f"embed {embed_bpw}bpw", "embed", r["weights_gib"]["embed"], emb)

    # ------------------------------------------------------------------
    # docs/02 §3.3 : context x KV-bits matrix on P1
    # ------------------------------------------------------------------
    ctx_kv = {
        4096: (6.45, 6.05, 5.84),
        8192: (7.27, 6.45, 6.05),
        16384: (8.89, 7.27, 6.45),
        32768: (12.14, 8.89, 7.27),
        65536: (18.64, 12.14, 8.89),
    }
    for ctx, vals in ctx_kv.items():
        for kv, want in zip((16, 8, 4), vals):
            r = estimate(p1(), Palette(name="x", bpw_dense=4.0, bpw_expert=4.0,
                                       bpw_embed=6.0, kv_bits=kv, ctx=ctx))
            au.eq(f"ctx{ctx}/kv{kv}", "vram", r["vram_gib"], want)

    # docs/02 §3.4 : KV bits effect on speed
    for kv, tps, decode_kv in ((16, 21.6, 0.42), (8, 51.1, 0.21), (4, 52.2, 0.11)):
        r = estimate(p1(), Palette(name="x", bpw_dense=4.0, bpw_expert=4.0,
                                   bpw_embed=6.0, kv_bits=kv, ctx=8192))
        au.eq(f"kv{kv} speed", "tps", r["decode_tok_s_ceiling"], tps, tol=0.06)
        au.eq(f"kv{kv} speed", "decode_kv", r["kv_decode_gib"], decode_kv)

    # ------------------------------------------------------------------
    # docs/02 §4.3 : MoE topology table
    # ------------------------------------------------------------------
    topo = [
        (12, 2, 4096, 1, 28.39, 8.25, 7.76, 4.82, 6.54, 9.4),
        (16, 2, 3072, 1, 27.88, 6.74, 7.58, 4.64, 6.54, 12.3),
        (16, 2, 2560, 1, 23.61, 5.99, 6.46, 4.56, 5.51, 14.5),
        (24, 2, 2048, 1, 27.38, 5.23, 7.41, 4.47, 6.54, 17.7),
        (24, 3, 2048, 1, 27.38, 6.24, 7.41, 4.47, 6.54, 13.8),
        (32, 2, 1792, 1, 31.28, 4.86, 8.39, 4.42, 7.57, 20.0),
        (32, 4, 1536, 1, 27.13, 5.99, 7.32, 4.38, 6.54, 14.7),
        (16, 4, 2048, 0, 18.32, 6.24, 4.99, 4.12, 4.47, 14.0),
        (24, 2, 1792, 2, 25.12, 5.74, 6.90, 4.73, 5.76, 15.3),
        (32, 2, 1536, 2, 27.88, 5.23, 7.58, 4.64, 6.54, 17.5),
    ]
    for E, topk, effn, sh, tot, act, w, vr, host, tps in topo:
        a = Arch(name="x", hidden=4096, layers=40, heads=32, kv_heads=8,
                 head_dim=128, ffn=14336, vocab=131072, tie_embed=True,
                 experts=E, experts_per_tok=topk, expert_ffn=effn,
                 shared_experts=sh, global_every=4, window=2048)
        r = estimate(a, Palette(name="x", bpw_dense=3.0, bpw_expert=2.2,
                                bpw_embed=4.5, kv_bits=8, ctx=32768))
        src = f"moe E{E}/k{topk}/{effn}/sh{sh}"
        au.eq(src, "params", r["params_total_b"], tot)
        au.eq(src, "active", r["params_active_b"], act)
        au.eq(src, "weights", r["weights_gib"]["total"], w)
        au.eq(src, "vram", r["vram_gib"], vr)
        au.eq(src, "host", r["host_gib"], host)
        au.eq(src, "tps", r["decode_tok_s_ceiling"], tps, tol=0.06)

    # ------------------------------------------------------------------
    # docs/02 §5 + ADR 0004 : bandwidth sensitivity
    # ------------------------------------------------------------------
    def with_bw(factory, key, **over):
        pal = Palette(**{**PALETTES[key].__dict__, **over})
        return estimate(factory(), pal)

    au.eq("bw p1 vram400", "tps",
          with_bw(p1, "p1", bw_vram_gbps=400.0)["decode_tok_s_ceiling"], 73.0, 0.15)
    au.eq("bw p1 host68", "tps",
          with_bw(p1, "p1", bw_host_gbps=68.0)["decode_tok_s_ceiling"], 51.1, 0.06)
    au.eq("bw p3 vram400", "tps",
          with_bw(p3, "p3", bw_vram_gbps=400.0)["decode_tok_s_ceiling"], 9.6, 0.06)
    au.eq("bw p3 host34", "tps",
          with_bw(p3, "p3", bw_host_gbps=34.0)["decode_tok_s_ceiling"], 17.3, 0.06)
    au.eq("bw p3 host68", "tps",
          with_bw(p3, "p3", bw_host_gbps=68.0)["decode_tok_s_ceiling"], 30.0, 0.06)
    au.eq("bw ref pcie22", "tps",
          with_bw(ref_27b, "ref", bw_pcie_gbps=22.0)["decode_tok_s_ceiling"], 1.5, 0.06)
    au.eq("bw p1 pcie22", "tps",
          with_bw(p1, "p1", bw_pcie_gbps=22.0)["decode_tok_s_ceiling"], 51.1, 0.06)

    # ------------------------------------------------------------------
    # docs/02 §1.1 : per-layer parameter decomposition
    # ------------------------------------------------------------------
    a = ref_27b()
    au.eq("params", "attn/layer M", a.attn_params_per_layer() / 1e6, 66.06, 0.006)
    au.eq("params", "ffn/layer M", a.dense_mlp_params() / 62 / 1e6, 346.82, 0.006)
    au.eq("params", "norm/layer M", a.norm_params() / 62 / 1e6, 0.011, 0.001)
    au.eq("params", "embed B", a.embed_params() / 1e9, 1.41, 0.006)
    au.eq("params", "total B", a.params() / 1e9, 27.01, 0.006)

    # docs/01 §4.2 : per-layer cost at 27B FFN width. A layer costs weights
    # PLUS its KV slice, which is what the doc table columns mean.
    per_layer = (a.attn_params_per_layer() + 3 * a.hidden * a.ffn + 2 * a.hidden)
    kv_per_layer_8k_kv8 = a.kv_per_layer_elems() * 8192 * 8 / 8 / GIB
    for bpw, want in ((3.0, 0.1442), (3.5, 0.1682), (4.0, 0.1923),
                      (4.5, 0.2163), (5.0, 0.2403)):
        au.eq(f"per-layer weights @{bpw}bpw", "GiB",
              per_layer * bpw / 8 / GIB, want, 0.0002)
    au.eq("per-layer", "2 layers weights+kv",
          2 * (per_layer * 4.0 / 8 / GIB + kv_per_layer_8k_kv8), 0.447, 0.002)

    # docs/01 §4.2 : P1-width per-layer cost, same weights+KV convention
    b = p1()
    p1_layer = (b.attn_params_per_layer() + 3 * b.hidden * b.ffn + 2 * b.hidden)
    au.eq("p1-per-layer", "weights @4.0", p1_layer * 4.0 / 8 / GIB, 0.169, 0.002)
    au.eq("p1-per-layer", "2 layers weights+kv",
          2 * (p1_layer * 4.0 / 8 / GIB + kv_per_layer_8k_kv8), 0.400, 0.002)
    au.eq("p1-bpw", "+0.5bpw", (b.params() - b.embed_params()) * 0.5 / 8 / GIB,
          0.550, 0.002)

    # docs/01 §4.4 : share of the reduction attributable to depth
    total_drop = 19.02 - 6.45
    au.eq("reduction", "total drop", total_drop, 12.57, 0.02)
    au.eq("reduction", "depth share pct", 10.04 / total_drop * 100, 79.9, 0.15)

    # docs/02 §1.3 : KV per token per layer
    k1 = Arch(name="x", hidden=5376, layers=1, heads=32, kv_heads=16,
              head_dim=128, ffn=21504, vocab=1024)
    au.eq("kv", "bytes/token/layer kv16", k1.kv_per_layer_elems() * 16 / 8, 8192,
          tol=0.6)
    for ctx, q, want in ((8192, 8, 0.0312), (8192, 16, 0.0625),
                         (32768, 8, 0.1250)):
        au.eq(f"kv/layer ctx{ctx} kv{q}", "GiB",
              k1.kv_per_layer_elems() * ctx * q / 8 / GIB, want, 0.0003)

    # docs/02 §3.2 : FFN narrowing and vocab chunk costs
    au.eq("ffn", "-2048 @4.0 GiB", 3 * 5376 * 2048 * 4.0 / 8 / GIB, 0.015, 0.001)
    au.eq("vocab", "64k @6bpw GiB", 65536 * 5376 * 6 / 8 / GIB, 0.246, 0.002)
    au.eq("vocab", "128k @6bpw GiB", 131072 * 5376 * 6 / 8 / GIB, 0.492, 0.002)

    # docs/03 §2.4 : routing changes decode KV only, never allocation
    for ge, want_decode in ((6, 0.210), (10, 0.160), (26, 0.130)):
        a2 = Arch(**{**p1().__dict__, "global_every": ge})
        r = estimate(a2, Palette(name="x", bpw_dense=4.0, bpw_expert=4.0,
                                 bpw_embed=6.0, kv_bits=8, ctx=8192))
        au.eq(f"routing ge{ge}", "decode_kv", r["kv_decode_gib"], want_decode, 0.002)
        au.eq(f"routing ge{ge}", "kv_alloc_unchanged", r["kv_gib"], 0.810, 0.002)

    # docs/BENCHMARKS.md + docs/02 §5 : CPU-vs-GPU limit arithmetic
    au.eq("gemv", "compute ms", 2 * 4096 * 4096 / 10.46e9 * 1e3, 3.21, 0.01)
    au.eq("gemv", "bandwidth ms", 4096 * 4096 * 4 / 280e9 * 1e3, 0.240, 0.005)
    au.eq("gemv", "ratio", (2 * 4096 * 4096 / 10.46e9) / (4096 * 4096 * 4 / 280e9),
          13.4, 0.1)
    au.eq("cpu matmul", "8.25B s", 2 * 8.25e9 / 10.46e9, 1.58, 0.01)

    # docs/05 : hardware arithmetic
    au.eq("hw", "ddr5-4800 1ch GB/s", 4800 * 8 / 1000, 38.4, 0.05)
    au.eq("hw", "single-stick efficiency pct", 19.3 / 38.4 * 100, 50.3, 0.5)
    au.eq("hw", "dual 2ch GB/s", 4800 * 8 * 2 / 1000, 76.8, 0.05)
    au.eq("hw", "python dequant gap", 19.3 / 0.8, 24.1, 0.5)
    au.eq("hw", "vram held by desktop MiB", 8192 - 7178, 1014, 0.5)
    au.eq("hw", "prune pct", (1 - 10.15 / 27.01) * 100, 62.4, 0.1)
    au.eq("hw", "p1 half of ceiling", 51.1 * 0.5, 25.55, 0.01)

    # ------------------------------------------------------------------
    # docs/06-hardware-tiers.md : GPU tier matrix
    # Budgeted bandwidth = 62% of the peak the doc lists.
    # ------------------------------------------------------------------
    TIER_BW_FACTOR = 0.62
    tiers = [
        # name, peak GB/s, p1, p1l, p2, p3
        ("4090 Laptop 16G", 576, 65.2, 86.3, 46.8, 9.6),
        ("4080 Laptop 12G", 432, 48.9, 64.7, 38.5, 9.3),
        ("3070Ti Laptop 8G", 448, 50.7, 67.1, 39.5, 9.4),
        ("3060 Laptop 6G", 336, 38.0, 50.4, 32.1, 9.1),
        ("4060 Laptop 8G", 272, 30.8, 40.8, 27.2, 8.9),
        ("3050 Laptop 8G", 224, 25.4, 33.6, 23.3, 8.7),
    ]
    for name, peak, w_p1, w_p1l, w_p2, w_p3 in tiers:
        bw = peak * TIER_BW_FACTOR
        for key, want in (("p1", w_p1), ("p1l", w_p1l), ("p2", w_p2), ("p3", w_p3)):
            pal = Palette(**{**PALETTES[key].__dict__, "bw_vram_gbps": bw})
            got = estimate(ARCHS[key](), pal)["decode_tok_s_ceiling"]
            au.eq(f"tier {name} {key}", "tps", got, want, tol=0.06)

    # docs/06 §1 : the headline "P1 falls below the floor on a 3050" claim
    r = estimate(p1(), Palette(**{**PALETTES["p1"].__dict__,
                                  "bw_vram_gbps": 224 * TIER_BW_FACTOR}))
    if r["decode_tok_s_ceiling"] >= 30.0:
        au.note("tier claim", "P1 on 3050 should be below the 30 tok/s floor, "
                              f"model says {r['decode_tok_s_ceiling']}")

    # docs/06 §5 : largest config that fits a 4 GiB card must still miss
    a4 = Arch(name="x", hidden=5376, layers=20, heads=32, kv_heads=16,
              head_dim=128, ffn=14336, vocab=65536, tie_embed=True,
              global_every=6, window=1024)
    r = estimate(a4, Palette(name="x", bpw_dense=3.5, bpw_expert=3.5,
                             bpw_embed=6.0, kv_bits=8, ctx=8192, vram_gib=3.2))
    au.eq("4GiB smallest", "params", r["params_total_b"], 6.30)
    au.eq("4GiB smallest", "vram", r["vram_gib"], 4.04)
    au.eq("4GiB smallest", "deficit", 3.2 - r["vram_gib"], -0.84)

    # ------------------------------------------------------------------
    # docs/01 §4.5 : P1 vs P1-low comparison table
    # ------------------------------------------------------------------
    p1row = estimate(p1(), Palette(**PALETTES["p1"].__dict__))
    lowrow = estimate(p1_low(), Palette(**PALETTES["p1l"].__dict__))
    au.eq("p1-low table", "p1 params", p1row["params_total_b"], 10.15)
    au.eq("p1-low table", "low params", lowrow["params_total_b"], 8.43)
    au.eq("p1-low table", "p1 weights", p1row["weights_gib"]["total"], 4.89)
    au.eq("p1-low table", "low weights", lowrow["weights_gib"]["total"], 3.64)
    au.eq("p1-low table", "p1 vram", p1row["vram_gib"], 6.45)
    au.eq("p1-low table", "low vram", lowrow["vram_gib"], 5.20)
    au.eq("p1-low table", "low headroom", 7.0 - lowrow["vram_gib"], 1.80)
    # docs/01 §4.5 quotes its row at 278 GB/s (the reference box), which is what
    # the README head table uses too. Keep both on the same basis so a reader
    # comparing the two tables does not see a phantom 0.4 tok/s difference.
    p1_278 = estimate(p1(), Palette(**{**PALETTES["p1"].__dict__,
                                       "bw_vram_gbps": 278.0}))
    low_278 = estimate(p1_low(), Palette(**{**PALETTES["p1l"].__dict__,
                                            "bw_vram_gbps": 278.0}))
    au.eq("p1-low table", "p1 tps @278", p1_278["decode_tok_s_ceiling"], 50.7,
          0.06)
    au.eq("p1-low table", "low tps @278", low_278["decode_tok_s_ceiling"], 67.2,
          0.06)
    # 16k context on P1-low still fits
    low16 = estimate(p1_low(), Palette(**{**PALETTES["p1l"].__dict__,
                                          "ctx": 16384}))
    au.eq("p1-low table", "low 16k vram", low16["vram_gib"], 6.02)
    au.eq("p1-low table", "low 16k headroom", 7.0 - low16["vram_gib"], 0.98)

    # ------------------------------------------------------------------
    # README.md head table : P1 vs P1-low across the three tiers it mentions
    # ------------------------------------------------------------------
    for gpu, bw, want_p1, want_low in (
            ("3070Ti (reference)", 278.0, 50.7, 67.2),
            ("4060", 169.0, 30.8, 40.9),
            ("3060", 208.0, 38.0, 50.3),
            ("3050", 139.0, 25.4, 33.6)):
        gp = estimate(p1(), Palette(**{**PALETTES["p1"].__dict__,
                                       "bw_vram_gbps": bw}))
        gl = estimate(p1_low(), Palette(**{**PALETTES["p1l"].__dict__,
                                           "bw_vram_gbps": bw}))
        au.eq(f"README head {gpu} p1", "tps", gp["decode_tok_s_ceiling"],
              want_p1, tol=0.06)
        au.eq(f"README head {gpu} p1-low", "tps", gl["decode_tok_s_ceiling"],
              want_low, tol=0.06)

    # ------------------------------------------------------------------
    # CLI contract: --explain must honour --arch
    # ------------------------------------------------------------------
    proc = subprocess.run(
        [sys.executable, "-I", str(ROOT / "tools" / "budget_model.py"),
         "--explain", "--arch", "p3", "--json"],
        capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        au.note("cli", f"--explain --arch p3 failed: {proc.stderr.strip()[:120]}")
    else:
        rows = json.loads(proc.stdout)
        if len(rows) != 1:
            au.note("cli", f"--explain --arch p3 returned {len(rows)} rows, want 1")

    print(f"audited {au.checked} documented numbers from 6 files")
    if au.fails:
        print(f"\n{len(au.fails)} DRIFTED:")
        for f in au.fails:
            print(f"  {f}")
        return 1
    print("all reproduced within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

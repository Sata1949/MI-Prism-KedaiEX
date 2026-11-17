#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Moss KedaiEX budget & throughput model.

Zero dependencies (stdlib only). Every number quoted in docs/ comes from this
script, so prose cannot drift from the model it describes.

What it answers
---------------
1. Does a config fit 8 GiB VRAM + 16 GiB RAM, and by how much does it miss?
2. What is the decode-speed ceiling on that hardware (memory-bandwidth bound)?
3. How much does each lever (layers, width, bpw, KV bits, sparsity) buy?

Units
-----
params_b    raw parameter count, billions
bpw         effective bits per weight of the packed file, metadata included
kvGiB       KV cache at the configured context
spill-GiB   GiB that must cross the PCIe bus per token because it lives in RAM
tok/s       ceiling = usable bandwidth / bytes touched per decoded token

Decode is modelled as bandwidth-bound, which is correct for batch=1 on consumer
GPUs. Prefill (compute-bound) is not modelled here; it is tracked separately in
the benchmark harness once real weights exist.

Usage
-----
    python tools/budget_model.py
    python tools/budget_model.py --json
    python tools/budget_model.py --arch p2 --set ctx=32768 --set bpw_dense=3.0
    python tools/budget_model.py --ablation
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict

GIB = 1024 ** 3
# Bandwidth and capacity constants, all measured on the reference box by
# tools/probe_hardware.py. Spec sheets are NOT used: they over-promise by 2x on
# laptop parts and the model would then claim configs fit that do not.
#
#   GPU    RTX 3070 Ti Laptop, 8 GiB GDDR6 256-bit @14 Gbps
#          peak 448 GB/s; budget 280 GB/s (62%) for a bandwidth-bound GEMV chain.
#   RAM    DDR5-5600 running at 4800 MT/s, single 16 GiB stick, Ryzen 7 6800H.
#          measured streaming read 18.8-20.5 GB/s across 3 runs (8 threads),
#          single-thread only 6.0-6.4 GB/s => any RAM-resident weight path
#          MUST be multi-threaded or it loses 3x. Budget 17 GB/s.
#   PCIe   Gen4 x8 (advertised max x16, negotiated x8 on this laptop).
#          15.75 GB/s theoretical; budget 11 GB/s sustained host<->device.
#
#   USABLE capacity, not nominal: a 16 GB machine reports 15.19 GiB and an
#   8 GiB card has 7.0 GiB free with the desktop compositor running. Reserving
#   for the OS leaves 7.0 GiB VRAM / 12.0 GiB RAM to actually run a model.
BW_PCIE_GBPS = 11.0
BW_HOST_GBPS = 17.0
BW_VRAM_GBPS = 280.0
USABLE_VRAM_GIB = 7.0
USABLE_RAM_GIB = 12.0


# --------------------------------------------------------------------------
# architecture
# --------------------------------------------------------------------------
@dataclass
class Arch:
    """Dense core + optional MoE expert bank. Attention is always dense."""

    name: str
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int                  # dense MLP width (unused when experts > 0)
    vocab: int
    tie_embed: bool = True
    experts: int = 0          # 0 => pure dense
    experts_per_tok: int = 0
    expert_ffn: int = 0
    shared_experts: int = 0
    global_every: int = 0     # 0 => full attention every layer
    window: int = 0
    kv_lora_rank: int = 0     # MLA latent rank; 0 => GQA/MHA

    # ---------------- params ----------------
    def attn_params_per_layer(self) -> int:
        h, kv, d = self.hidden, self.kv_heads, self.head_dim
        return h * self.heads * d + 2 * h * kv * d + self.heads * d * h

    def mla_params_per_layer(self) -> int:
        h, c = self.hidden, self.kv_lora_rank
        return (h * self.heads * self.head_dim
                + h * c
                + c * (self.heads + self.kv_heads) * self.head_dim
                + self.heads * self.head_dim * h)

    def attn_layer_params(self) -> int:
        return (self.mla_params_per_layer() if self.kv_lora_rank
                else self.attn_params_per_layer())

    def embed_params(self) -> int:
        return self.vocab * self.hidden * (1 if self.tie_embed else 2)

    def expert_mlp_params(self) -> int:
        if not self.experts:
            return 0
        return self.layers * self.experts * 3 * self.hidden * self.expert_ffn

    def shared_mlp_params(self) -> int:
        if not self.shared_experts:
            return 0
        return (self.layers * self.shared_experts * 3
                * self.hidden * self.expert_ffn)

    def dense_mlp_params(self) -> int:
        if self.experts:
            return 0
        return self.layers * 3 * self.hidden * self.ffn

    def norm_params(self) -> int:
        return 2 * self.hidden * self.layers

    def params(self) -> int:
        return (self.layers * self.attn_layer_params()
                + self.embed_params()
                + self.norm_params()
                + self.expert_mlp_params()
                + self.shared_mlp_params()
                + self.dense_mlp_params())

    def active_params(self) -> int:
        """Params read for one decoded token."""
        if not self.experts:
            return self.params()
        per_expert = 3 * self.hidden * self.expert_ffn
        active_mlp = (self.experts_per_tok + self.shared_experts) * per_expert
        return (self.layers * self.attn_layer_params()
                + self.embed_params()
                + self.norm_params()
                + self.layers * active_mlp)

    # ---------------- KV ----------------
    def kv_per_layer_elems(self) -> int:
        if self.kv_lora_rank:
            return 2 * self.kv_lora_rank
        return 2 * self.kv_heads * self.head_dim

    def kv_gib(self, ctx: int, batch: int = 1, kv_bits: int = 16) -> float:
        n = self.layers * self.kv_per_layer_elems() * ctx * batch
        return n * kv_bits / 8.0 / GIB

    def decode_kv_gib(self, ctx: int, batch: int = 1, kv_bits: int = 16) -> float:
        """KV actually read per decoded token with local/global interleave."""
        if not self.global_every:
            return self.kv_gib(ctx, batch, kv_bits)
        local = sum(1 for n in range(self.layers) if (n + 1) % self.global_every)
        glob = self.layers - local
        tokens = glob * ctx + local * min(ctx, self.window)
        return (tokens * self.kv_per_layer_elems() * kv_bits / 8.0
                * batch / GIB)


@dataclass
class Palette:
    """Quant mix + runtime + hardware envelope."""

    name: str = "custom"
    bpw_dense: float = 4.5    # attention + norms + dense MLP
    bpw_expert: float = 2.2   # MoE experts (streamed from RAM)
    bpw_embed: float = 6.0    # embedding / head, kept high
    kv_bits: int = 8
    ctx: int = 8192
    batch: int = 1
    compute_buf_gib: float = 0.40   # activations + masking at batch 1
    runtime_overhead_gib: float = 0.35  # CUDA ctx, allocator, graph capture
    vram_gib: float = USABLE_VRAM_GIB
    ram_gib: float = USABLE_RAM_GIB
    spill_budget_gib: float = 2.0   # accepted PCIe traffic per token
    # Bandwidth overrides. None means "use the measured module constant".
    # Exposed here so a different machine (or a hypothetical upgrade such as a
    # second DIMM) can be modelled without editing source.
    bw_vram_gbps: float | None = None
    bw_host_gbps: float | None = None
    bw_pcie_gbps: float | None = None

    def bandwidth(self) -> tuple[float, float, float]:
        return (
            self.bw_vram_gbps if self.bw_vram_gbps is not None else BW_VRAM_GBPS,
            self.bw_host_gbps if self.bw_host_gbps is not None else BW_HOST_GBPS,
            self.bw_pcie_gbps if self.bw_pcie_gbps is not None else BW_PCIE_GBPS,
        )


def estimate(arch: Arch, pal: Palette) -> dict:
    p_total = arch.params()
    p_expert = arch.expert_mlp_params()
    p_shared = arch.shared_mlp_params()
    p_embed = arch.embed_params()
    p_dense = p_total - p_expert - p_embed

    w_dense = p_dense * pal.bpw_dense / 8 / GIB
    w_expert = p_expert * pal.bpw_expert / 8 / GIB
    w_embed = p_embed * pal.bpw_embed / 8 / GIB
    w_total = w_dense + w_expert + w_embed

    bw_vram, bw_host, bw_pcie = pal.bandwidth()

    kv_total = arch.kv_gib(pal.ctx, pal.batch, pal.kv_bits)
    kv_decode = arch.decode_kv_gib(pal.ctx, pal.batch, pal.kv_bits)

    # Placement: dense weights + KV + buffers in VRAM; expert bank in RAM.
    vram = w_dense + w_embed + kv_total + pal.compute_buf_gib + pal.runtime_overhead_gib
    host = w_expert + pal.runtime_overhead_gib
    spill = max(0.0, vram - pal.vram_gib)
    host_ok = host <= pal.ram_gib

    # Decode bytes per token, split by where they live. Dense weights and the
    # decode-time KV slice stay in VRAM; routed experts are streamed from RAM.
    dense_bytes = (w_dense + w_embed + kv_decode) * GIB
    if arch.experts:
        expert_bytes = (3 * arch.hidden * arch.expert_ffn * pal.bpw_expert / 8.0
                        * arch.layers
                        * (arch.experts_per_tok + arch.shared_experts))
    else:
        expert_bytes = 0.0

    t_vram = dense_bytes / (bw_vram * 1e9)
    t_spill = spill * GIB / (bw_pcie * 1e9)
    t_host = expert_bytes / (bw_host * 1e9) if arch.experts else 0.0
    t_total = t_vram + t_spill + t_host
    tok_s = 1.0 / max(t_total, 1e-9)

    if spill == 0:
        fit = "resident"
    elif spill <= pal.spill_budget_gib:
        fit = f"partial (+{spill:.2f} GiB over PCIe)"
    else:
        fit = f"over budget (+{spill:.2f} GiB)"

    return {
        "arch": arch.name,
        "palette": pal.name,
        "params_total_b": round(p_total / 1e9, 2),
        "params_active_b": round(arch.active_params() / 1e9, 2),
        "params_expert_b": round(p_expert / 1e9, 2),
        "weights_gib": {
            "dense": round(w_dense, 2),
            "expert": round(w_expert, 2),
            "embed": round(w_embed, 2),
            "total": round(w_total, 2),
            "effective_bpw": round(w_total * 8 * GIB / p_total, 2),
        },
        "kv_gib": round(kv_total, 2),
        "kv_decode_gib": round(kv_decode, 2),
        "ctx": pal.ctx,
        "vram_gib": round(vram, 2),
        "vram_cap_gib": pal.vram_gib,
        "spill_gib": round(spill, 2),
        "host_gib": round(host, 2),
        "host_cap_gib": pal.ram_gib,
        "host_fit": host_ok,
        "placement": fit,
        "decode_tok_s_ceiling": round(tok_s, 1),
        "bandwidth_gbps": {
            "vram": bw_vram, "host": bw_host, "pcie": bw_pcie,
        },
        # Time decomposition, ms per token. Sums to 1000/tok_s.
        "ms_per_token": {
            "vram_resident": round(t_vram * 1e3, 1),
            "pcie_spill": round(t_spill * 1e3, 1),
            "host_stream": round(t_host * 1e3, 1),
            "total": round(t_total * 1e3, 1),
        },
        "bytes_per_token_mb": {
            "vram": round(dense_bytes / 1e6, 1),
            "host": round(expert_bytes / 1e6, 1),
        },
        # What you would get if host streaming were fully hidden behind GPU
        # compute (phase-3 prefetch target). Not achievable for dense configs:
        # their spill bytes travel over PCIe, which is already in this sum.
        "tok_s_if_host_overlap_perfect": round(
            1.0 / max(t_vram + t_spill, 1e-9), 1),
    }


# --------------------------------------------------------------------------
# configurations
# --------------------------------------------------------------------------
def ref_27b() -> Arch:
    return Arch(name="27B reference (62L, hybrid 5:1, GQA 32/16)",
                hidden=5376, layers=62, heads=32, kv_heads=16, head_dim=128,
                ffn=21504, vocab=262144, tie_embed=True,
                global_every=6, window=1024)


def p1() -> Arch:
    """Phase 1 shipped target. Must match configs/p1_target.json exactly.

    Three levers, all weight-preserving (no retraining of the trunk):
      depth 62 -> 26   selected by layer importance, not uniform stride
      ffn  21504 -> 18432
      vocab 262144 -> 131072
    """
    return Arch(name="P1 target (26L, ffn 18432, vocab 128k)",
                hidden=5376, layers=26, heads=32, kv_heads=16, head_dim=128,
                ffn=18432, vocab=131072, tie_embed=True,
                global_every=6, window=1024)


def p1_low() -> Arch:
    """Phase 1 low-end target. Must match configs/p1_low.json exactly.

    Narrower FFN and one step lower precision than p1(), which buys enough
    bandwidth headroom to clear 30 tok/s on a 224 GB/s-peak laptop GPU
    (RTX 3050/3060 class). p1() does not: it lands at 25.4 tok/s there.
    """
    return Arch(name="P1-low target (26L, ffn 14336, vocab 128k)",
                hidden=5376, layers=26, heads=32, kv_heads=16, head_dim=128,
                ffn=14336, vocab=131072, tie_embed=True,
                global_every=6, window=1024)


def p1_depth_only() -> Arch:
    """Ablation anchor: trim depth only, leave FFN and vocab at 27B values."""
    return Arch(name="P1 depth-only (26L, ffn 21504, vocab 256k)",
                hidden=5376, layers=26, heads=32, kv_heads=16, head_dim=128,
                ffn=21504, vocab=262144, tie_embed=True,
                global_every=6, window=1024)


def p1_fallback() -> Arch:
    """If 26L loses too much quality: keep 32L, still fits nothing. Documented
    as the rejected branch so nobody re-proposes it."""
    return Arch(name="P1 fallback (32L, ffn 21504, vocab 128k)",
                hidden=5376, layers=32, heads=32, kv_heads=16, head_dim=128,
                ffn=21504, vocab=131072, tie_embed=True,
                global_every=6, window=1024)


def p2() -> Arch:
    return Arch(name="P2 width-cut (~9.3B, 40L, GQA 32/8)",
                hidden=4096, layers=40, heads=32, kv_heads=8, head_dim=128,
                ffn=14336, vocab=131072, tie_embed=True,
                global_every=4, window=2048)


def p3() -> Arch:
    return Arch(name="P3 MoE (~28B total / 8B active)",
                hidden=4096, layers=40, heads=32, kv_heads=8, head_dim=128,
                ffn=14336, vocab=131072, tie_embed=True,
                experts=12, experts_per_tok=2, expert_ffn=4096, shared_experts=1,
                global_every=4, window=2048)


PALETTES = {
    "ref": Palette(name="4.5bpw/KV16/8k", bpw_dense=4.5, bpw_expert=4.5,
                   bpw_embed=6.0, kv_bits=16, ctx=8192),
    "p1":  Palette(name="4.0bpw/KV8/8k [SHIPPED]", bpw_dense=4.0,
                   bpw_expert=4.0, bpw_embed=6.0, kv_bits=8, ctx=8192),
    "p1l": Palette(name="3.5bpw/KV8/8k [LOW-END]", bpw_dense=3.5,
                   bpw_expert=3.5, bpw_embed=6.0, kv_bits=8, ctx=8192),
    "p1d": Palette(name="4.0bpw/KV8/8k", bpw_dense=4.0, bpw_expert=4.0,
                   bpw_embed=6.0, kv_bits=8, ctx=8192),
    "p1f": Palette(name="4.5bpw/KV8/8k", bpw_dense=4.5, bpw_expert=4.5,
                   bpw_embed=6.0, kv_bits=8, ctx=8192),
    "p2":  Palette(name="3.4bpw/KV8/32k", bpw_dense=3.4, bpw_expert=3.2,
                   bpw_embed=6.0, kv_bits=8, ctx=32768),
    "p3":  Palette(name="3.0bpw/KV8/32k", bpw_dense=3.0, bpw_expert=2.2,
                   bpw_embed=4.5, kv_bits=8, ctx=32768),
}

ARCHS = {
    "ref": ref_27b,
    "p1": p1,
    "p1l": p1_low,
    "p1d": p1_depth_only,
    "p1f": p1_fallback,
    "p2": p2,
    "p3": p3,
}

# One lever at a time, measured from the P1 baseline (26L / ffn 18432 /
# vocab 131072 / 4.0bpw / KV8 / 8k ctx). Reported by --ablation.
ABLATIONS = [
    ("baseline P1", {}, {}),
    ("vocab 128k -> 64k", {}, {"vocab": 65536}),
    ("vocab 128k -> 32k", {}, {"vocab": 32768}),
    ("depth 26 -> 24", {}, {"layers": 24}),
    ("depth 26 -> 22", {}, {"layers": 22}),
    ("depth 26 -> 28", {}, {"layers": 28}),
    ("depth 26 -> 32", {}, {"layers": 32}),
    ("ffn 18432 -> 21504", {}, {"ffn": 21504}),
    ("ffn 18432 -> 16384", {}, {"ffn": 16384}),
    ("ffn 18432 -> 14336", {}, {"ffn": 14336}),
    ("bpw 4.0 -> 4.5", {"bpw_dense": 4.5, "bpw_expert": 4.5}, {}),
    ("bpw 4.0 -> 5.0", {"bpw_dense": 5.0, "bpw_expert": 5.0}, {}),
    ("bpw 4.0 -> 3.5", {"bpw_dense": 3.5, "bpw_expert": 3.5}, {}),
    ("kv 8 -> 16 bit", {"kv_bits": 16}, {}),
    ("kv 8 -> 4 bit", {"kv_bits": 4}, {}),
    ("ctx 8k -> 16k", {"ctx": 16384}, {}),
    ("ctx 8k -> 32k", {"ctx": 32768}, {}),
]


def coerce_field(pal: Palette | object, key: str, raw: str) -> object:
    """Parse a --set value to the type the dataclass field wants.

    Fields whose current value is None (the bandwidth overrides) are floats by
    declaration, so fall back to the annotation rather than the value.
    """
    if not hasattr(pal, key):
        raise SystemExit(f"unknown palette field: {key}")
    cur = getattr(pal, key)
    if isinstance(cur, bool):
        raise SystemExit(f"field {key} is boolean; not settable from CLI")
    if isinstance(cur, int):
        return int(raw)
    if isinstance(cur, float):
        return float(raw)
    # None-valued override field: annotation decides
    ann = type(pal).__annotations__.get(key)
    if ann in (float, "float | None", "float"):
        return float(raw)
    if ann in (int, "int | None", "int"):
        return int(raw)
    return float(raw)


def _apply(pal: Palette, arch_kwargs: dict, overrides: list[str]) -> tuple:
    arch_kwargs = dict(arch_kwargs)
    for item in overrides:
        k, _, v = item.partition("=")
        setattr(pal, k, coerce_field(pal, k, v))
    return pal, arch_kwargs


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--arch", default="all", choices=["all"] + list(ARCHS))
    ap.add_argument("--set", action="append", default=[], metavar="K=V",
                    help="override a palette field, e.g. --set ctx=32768")
    ap.add_argument("--all-palettes", action="store_true",
                    help="cross every arch with every palette")
    ap.add_argument("--ablation", action="store_true",
                    help="one-lever-at-a-time report around the P1 baseline")
    ap.add_argument("--explain", action="store_true",
                    help="per-token time and byte breakdown for the selection")
    args = ap.parse_args(argv)

    if args.explain:
        rows = []
        for key, factory in ARCHS.items():
            if args.arch not in ("all", key):
                continue
            pal = Palette(**asdict(PALETTES[key]))
            pal, _ = _apply(pal, {}, args.set)
            r = estimate(factory(), pal)
            r["lever"] = key
            rows.append(r)
        if args.json:
            json.dump(rows, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
            return 0
        for r in rows:
            m = r["ms_per_token"]
            b = r["bytes_per_token_mb"]
            print(f"\n== {r['arch']}  [{r['palette']}] ==")
            print(f"   params      {r['params_total_b']:>6.2f} B total / "
                  f"{r['params_active_b']:.2f} B active")
            print(f"   weights     {r['weights_gib']['total']:>6.2f} GiB  "
                  f"(dense {r['weights_gib']['dense']}, "
                  f"embed {r['weights_gib']['embed']}, "
                  f"expert {r['weights_gib']['expert']})")
            print(f"   kv cache    {r['kv_gib']:>6.2f} GiB total, "
                  f"{r['kv_decode_gib']:.2f} GiB read per token")
            print(f"   vram        {r['vram_gib']:>6.2f} / {r['vram_cap_gib']} GiB"
                  f"   host {r['host_gib']:.2f} / {r['host_cap_gib']} GiB")
            print(f"   per token   {b['vram']:>8.1f} MB from VRAM, "
                  f"{b['host']:.1f} MB from host")
            print(f"   time        {m['vram_resident']:>7.1f} ms VRAM + "
                  f"{m['pcie_spill']:.1f} ms PCIe + "
                  f"{m['host_stream']:.1f} ms host = {m['total']:.1f} ms")
            print(f"   ceiling     {r['decode_tok_s_ceiling']:.1f} tok/s"
                  f"  (with perfect host overlap: "
                  f"{r['tok_s_if_host_overlap_perfect']:.1f} tok/s)")
        return 0

    if args.ablation:
        base = p1()
        base_kwargs = {k: getattr(base, k) for k in
                       ("name", "hidden", "layers", "heads", "kv_heads",
                        "head_dim", "ffn", "vocab", "tie_embed",
                        "global_every", "window")}
        rows = []
        for label, pal_over, arch_over in ABLATIONS:
            kw = dict(base_kwargs)
            kw.update(arch_over)
            kw["name"] = "x"
            arch = Arch(**kw)
            pal = Palette(**asdict(PALETTES["p1"]))
            for k, v in pal_over.items():
                setattr(pal, k, v)
            pal, _ = _apply(pal, {}, args.set)
            r = estimate(arch, pal)
            r["lever"] = label
            r["headroom_gib"] = round(pal.vram_gib - r["vram_gib"], 2)
            rows.append(r)

        if args.json:
            json.dump(rows, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
            return 0

        fmt = ("{label:24} {p:>7} {w:>6} {kv:>6} {vram:>6} {hr:>7} "
               "{tps:>6}  {place}")
        print(fmt.format(label="lever", p="params", w="wGiB", kv="kvGiB",
                         vram="vram", hr="headroom", tps="tok/s",
                         place="placement"))
        print("-" * 104)
        for r in rows:
            print(fmt.format(
                label=r["lever"], p=f"{r['params_total_b']:.2f}",
                w=f"{r['weights_gib']['total']:.2f}", kv=f"{r['kv_gib']:.2f}",
                vram=f"{r['vram_gib']:.2f}", hr=f"{r['headroom_gib']:+.2f}",
                tps=f"{r['decode_tok_s_ceiling']:.0f}",
                place=r["placement"]))
        return 0

    rows = []
    for key, factory in ARCHS.items():
        if args.arch not in ("all", key):
            continue
        pal_keys = list(PALETTES) if args.all_palettes else [key]
        for pk in pal_keys:
            pal = Palette(**asdict(PALETTES[pk]))
            pal, _ = _apply(pal, {}, args.set)
            rows.append(estimate(factory(), pal))

    if args.json:
        json.dump(rows, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0

    fmt = ("{arch:40} {pal:22} {p:>6} {a:>6} {w:>5} {kv:>5} "
           "{vram:>5} {host:>5} {tps:>5}  {place}")
    print(fmt.format(arch="config", pal="palette", p="params", a="active",
                     w="wGiB", kv="kvGiB", vram="vram", host="host",
                     tps="tok/s", place="placement"))
    print("-" * 146)
    for r in rows:
        print(fmt.format(
            arch=r["arch"], pal=r["palette"], p=f"{r['params_total_b']:.1f}",
            a=f"{r['params_active_b']:.1f}", w=f"{r['weights_gib']['total']:.2f}",
            kv=f"{r['kv_gib']:.2f}", vram=f"{r['vram_gib']:.2f}",
            host=f"{r['host_gib']:.2f}", tps=f"{r['decode_tok_s_ceiling']:.0f}",
            place=r["placement"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

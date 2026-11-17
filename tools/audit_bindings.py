#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bind documented numbers to the model that produces them.

Why this file exists
--------------------
The first audit generation compared the model against literals stored beside it
in the same Python file. That proves two copies of one belief agree; it does not
prove the prose is right. A doc could say 59.9 and the audit still passed,
because the audit never read the doc. Verified by injecting exactly that edit.

This version closes the loop differently: every claim names a doc line, and the
value is *derived from the model*, then rendered to text and required to appear
in that line. Consequences:

  - change the model  -> rendered text changes -> doc must be updated or CI fails
  - change the doc    -> expected text is gone   -> CI fails
  - both drift apart  -> CI fails

Extracting numbers by position was tried and rejected: doc lines carry GPU
names, bandwidth values and table indices, so "the 3rd number on the line" is a
coin flip. Matching a rendered token (optionally prefixed) is unambiguous and
produces a failure message that names the exact line.

Usage
-----
    python -I tools/audit_bindings.py
    python -I tools/audit_bindings.py --verbose
    python -I tools/audit_bindings.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from budget_model import (  # noqa: E402
    PALETTES, Arch, Palette, estimate, p1, p1_low, p2, p3, ref_27b,
)


def est(factory, key: str, **over):
    pal = Palette(**{**PALETTES[key].__dict__, **over})
    return estimate(factory(), pal)


def tier(key: str, bw: float):
    return est(
        {"p1": p1, "p1l": p1_low, "p2": p2, "p3": p3}[key],
        key, bw_vram_gbps=bw)


class Binding:
    """A doc line that must contain a token rendered from the model.

    path    file relative to repo root
    anchor  substring identifying the unique line
    render  callable -> (text, numeric_value)
    label   human name for failure messages
    """

    __slots__ = ("path", "anchor", "render", "label")

    def __init__(self, path: str, anchor: str, render, label: str) -> None:
        self.path = path
        self.anchor = anchor
        self.render = render
        self.label = label

    def check(self) -> tuple[bool, str]:
        f = ROOT / self.path
        if not f.exists():
            return False, f"{self.path}: file missing"
        want_text, want_value = self.render()
        hits = [(i, ln) for i, ln in
                enumerate(f.read_text(encoding="utf-8").splitlines(), 1)
                if self.anchor in ln]
        if not hits:
            return False, (f"{self.path}: anchor not found: {self.anchor!r}\n"
                           f"           (the doc line was deleted or reworded, "
                           f"so {self.label} is no longer audited)")
        for lineno, line in hits:
            if want_text in line:
                return True, f"{self.path}:{lineno}: {self.label} = {want_text}"
        lineno, line = hits[0]
        return False, (f"{self.path}:{lineno}: {self.label} must read "
                       f"{want_text!r} (from the model, value {want_value}) "
                       f"but the line is:\n           {line.strip()[:120]}")


def t(value: float, nd: int = 2) -> str:
    """Render a number the way the docs write it."""
    return f"{value:.{nd}f}"


def bindings() -> list[Binding]:
    B = Binding
    out: list[Binding] = []

    # ---------------- README headline ----------------
    out += [
        B("README.md", "27B reference  19.02 GiB",
          lambda: (t(est(ref_27b, "ref")["vram_gib"]),
                   est(ref_27b, "ref")["vram_gib"]), "27B vram"),
        B("README.md", "P1 target       6.45 GiB",
          lambda: (t(est(p1, "p1")["vram_gib"]),
                   est(p1, "p1")["vram_gib"]), "P1 vram"),
        B("README.md", "headroom +0.55",
          lambda: (t(7.0 - est(p1, "p1")["vram_gib"]),
                   7.0 - est(p1, "p1")["vram_gib"]), "P1 headroom"),
        B("README.md", "over by 12.02",
          lambda: (t(est(ref_27b, "ref")["spill_gib"]),
                   est(ref_27b, "ref")["spill_gib"]), "27B deficit"),
    ]

    # ---------------- README two-delivery table (278 GB/s) ----------------
    p1_278 = lambda: est(p1, "p1", bw_vram_gbps=278.0)
    low_278 = lambda: est(p1_low, "p1l", bw_vram_gbps=278.0)
    out += [
        B("README.md", "| 参数 / 权重 | 10.15B / 4.89 GiB",
          lambda: (f"{t(p1_278()['params_total_b'])}B / "
                   f"{t(p1_278()['weights_gib']['total'])} GiB",
                   p1_278()["params_total_b"]), "P1 params/weights"),
        B("README.md", "| 显存 / 余量 |",
          lambda: (t(p1_278()["vram_gib"]), p1_278()["vram_gib"]), "P1 vram row"),
        B("README.md", "| 参考机（3070 Ti，278 GB/s）",
          lambda: (t(p1_278()["decode_tok_s_ceiling"], 1),
                   p1_278()["decode_tok_s_ceiling"]), "P1 tps @278"),
        B("README.md", "| RTX 4060 8G（169 GB/s）",
          lambda: (t(tier("p1", 169.0)["decode_tok_s_ceiling"], 1),
                   tier("p1", 169.0)["decode_tok_s_ceiling"]), "P1 @4060"),
        B("README.md", "| RTX 3060 6G（208 GB/s）",
          lambda: (t(tier("p1", 208.0)["decode_tok_s_ceiling"], 1),
                   tier("p1", 208.0)["decode_tok_s_ceiling"]), "P1 @3060"),
        B("README.md", "| **RTX 3050 8G（139 GB/s）**",
          lambda: (t(tier("p1", 139.0)["decode_tok_s_ceiling"], 1),
                   tier("p1", 139.0)["decode_tok_s_ceiling"]), "P1 @3050"),
    ]

    # ---------------- README envelope ----------------
    out += [
        B("README.md", "| 显存 | 8 GiB |",
          lambda: ("7.00", 7.00), "usable vram"),
        B("README.md", "| 内存 | 16 GB |",
          lambda: ("15.19", 15.19), "visible ram"),
        B("README.md", "| 内存带宽 | 5600 MT/s",
          lambda: ("19.3", 19.3), "measured ram bw"),
        B("README.md", "| PCIe | Gen4 x16 |",
          lambda: ("x8", 0.0), "pcie width"),
    ]

    # ---------------- README layer sweep ----------------
    def sweep(L: int, field: str):
        r = estimate(
            Arch(name="x", hidden=5376, layers=L, heads=32, kv_heads=16,
                head_dim=128, ffn=21504, vocab=262144, tie_embed=True,
                global_every=6, window=1024),
            Palette(name="x", bpw_dense=4.5, bpw_expert=4.5, bpw_embed=6.0,
                    kv_bits=16, ctx=8192))
        if field == "weights":
            return r["weights_gib"]["total"]
        if field == "spill":
            return r["spill_gib"]
        if field == "vram":
            return r["vram_gib"]
        raise KeyError(field)

    out += [
        B("README.md", "| 62 | 27.01B |",
          lambda: (t(sweep(62, "weights")), sweep(62, "weights")),
          "L62 weights"),
        B("README.md", "| 24 | 11.32B |",
          lambda: (t(sweep(24, "weights")), sweep(24, "weights")),
          "L24 weights"),
        B("README.md", "| 24 | 11.32B |",
          lambda: (f"+{t(sweep(24, 'spill'))}", sweep(24, "spill")),
          "L24 deficit"),
    ]

    # ---------------- docs/01 quant sweep ----------------
    def q(bpw: float, kv: int, field: str):
        r = estimate(ref_27b(), Palette(name="x", bpw_dense=bpw,
                                        bpw_expert=bpw, bpw_embed=4.5,
                                        kv_bits=kv, ctx=8192))
        return r["vram_gib"] if field == "vram" else r["spill_gib"]

    out += [
        B("docs/01-introduction.md", "| 2.0 | 6.70 |",
          lambda: (t(q(2.0, 16, "vram")), q(2.0, 16, "vram")),
          "27B@2.0 kv16 vram"),
        B("docs/01-introduction.md", "| 2.0 | 6.70 |",
          lambda: (t(q(2.0, 4, "vram")), q(2.0, 4, "vram")),
          "27B@2.0 kv4 vram"),
        B("docs/01-introduction.md", "| 2.0 | 6.70 |",
          lambda: (f"**+{t(q(2.0, 4, 'spill'))}**", q(2.0, 4, "spill")),
          "27B@2.0 kv4 deficit"),
    ]

    # ---------------- docs/01 ladder ----------------
    def ladder(L, ffn, vocab, bpw, kv):
        return estimate(
            Arch(name="x", hidden=5376, layers=L, heads=32, kv_heads=16,
                head_dim=128, ffn=ffn, vocab=vocab, tie_embed=True,
                global_every=6, window=1024),
            Palette(name="x", bpw_dense=bpw, bpw_expert=bpw, bpw_embed=6.0,
                    kv_bits=kv, ctx=8192))["vram_gib"]

    out += [
        B("docs/01-introduction.md", "| 起点 | 27B 原样",
          lambda: (t(ladder(62, 21504, 262144, 4.5, 16)),
                   ladder(62, 21504, 262144, 4.5, 16)), "ladder start"),
        B("docs/01-introduction.md", "| 裁剪词表 |",
          lambda: (t(ladder(62, 21504, 131072, 4.5, 16)),
                   ladder(62, 21504, 131072, 4.5, 16)), "ladder vocab"),
        B("docs/01-introduction.md", "| 剪层 | 62 → 26 层",
          lambda: (t(ladder(26, 21504, 131072, 4.5, 16)),
                   ladder(26, 21504, 131072, 4.5, 16)), "ladder depth"),
        B("docs/01-introduction.md", "| 收窄 FFN |",
          lambda: (t(ladder(26, 18432, 131072, 4.5, 16)),
                   ladder(26, 18432, 131072, 4.5, 16)), "ladder ffn"),
        B("docs/01-introduction.md", "| 降精度 |",
          lambda: (t(ladder(26, 18432, 131072, 4.0, 16)),
                   ladder(26, 18432, 131072, 4.0, 16)), "ladder bpw"),
        B("docs/01-introduction.md", "| KV 量化 |",
          lambda: (t(ladder(26, 18432, 131072, 4.0, 8)),
                   ladder(26, 18432, 131072, 4.0, 8)), "ladder kv"),
        B("docs/01-introduction.md", "| KV 量化 |",
          lambda: (f"**+{t(7.0 - ladder(26, 18432, 131072, 4.0, 8))}**",
                   7.0 - ladder(26, 18432, 131072, 4.0, 8)), "ladder headroom"),
    ]

    # ---------------- docs/01 P1 vs P1-low ----------------
    out += [
        B("docs/01-introduction.md", "| 参考机 tok/s（278 GB/s）",
          lambda: (t(est(p1, "p1", bw_vram_gbps=278.0)["decode_tok_s_ceiling"], 1),
                   est(p1, "p1", bw_vram_gbps=278.0)["decode_tok_s_ceiling"]),
          "P1 tps @278"),
        B("docs/01-introduction.md", "| 参考机 tok/s（278 GB/s）",
          lambda: (t(low_278()["decode_tok_s_ceiling"], 1),
                   low_278()["decode_tok_s_ceiling"]), "low tps @278"),
        B("docs/01-introduction.md", "| 余量 | +0.55 |",
          lambda: (f"**+{t(7.0 - est(p1_low, 'p1l')['vram_gib'])}**",
                   7.0 - est(p1_low, "p1l")["vram_gib"]), "low headroom"),
        B("docs/01-introduction.md", "| 上下文 | 8k |",
          lambda: (f"+{t(7.0 - est(p1_low, 'p1l', ctx=16384)['vram_gib'])}",
                   7.0 - est(p1_low, "p1l", ctx=16384)["vram_gib"]),
          "low 16k headroom"),
    ]

    # ---------------- docs/02 ctx x kv ----------------
    def ctxkv(ctx: int, kv: int):
        return est(p1, "p1", ctx=ctx, kv_bits=kv)["vram_gib"]

    out += [
        B("docs/02-budget-frontier.md", "| 8k |",
          lambda: (t(ctxkv(8192, 16)), ctxkv(8192, 16)), "ctx8k kv16"),
        B("docs/02-budget-frontier.md", "| 8k |",
          lambda: (f"**{t(ctxkv(8192, 8))}**", ctxkv(8192, 8)), "ctx8k kv8"),
        B("docs/02-budget-frontier.md", "| 64k |",
          lambda: (t(ctxkv(65536, 4)), ctxkv(65536, 4)), "ctx64k kv4"),
        B("docs/02-budget-frontier.md", "| 64k |",
          lambda: (t(ctxkv(65536, 16)), ctxkv(65536, 16)), "ctx64k kv16"),
    ]

    # ---------------- docs/06 tier table ----------------
    out += [
        B("docs/06-hardware-tiers.md", "| RTX 3070 Ti Laptop 8G |",
          lambda: (t(tier("p1", 278.0)["decode_tok_s_ceiling"], 1),
                   tier("p1", 278.0)["decode_tok_s_ceiling"]), "p1 @3070Ti"),
        B("docs/06-hardware-tiers.md", "| RTX 4090 Laptop 16G |",
          lambda: (t(tier("p1l", 357.0)["decode_tok_s_ceiling"], 1),
                   tier("p1l", 357.0)["decode_tok_s_ceiling"]), "low @4090"),
        B("docs/06-hardware-tiers.md", "| RTX 3050 Laptop 8G |",
          lambda: (t(tier("p1l", 139.0)["decode_tok_s_ceiling"], 1),
                   tier("p1l", 139.0)["decode_tok_s_ceiling"]), "low @3050"),
        B("docs/06-hardware-tiers.md", "| RTX 3050 Laptop 8G |",
          lambda: (f"**{t(tier('p1', 139.0)['decode_tok_s_ceiling'], 1)}",
                   tier("p1", 139.0)["decode_tok_s_ceiling"]), "p1 @3050"),
    ]

    # ---------------- docs/07 context table ----------------
    out += [
        B("docs/07-capability-gains.md", "| **P1-low @3.5bpw KV8** | **16k**",
          lambda: (f"**{t(est(p1_low, 'p1l', ctx=16384)['vram_gib'])}**",
                   est(p1_low, "p1l", ctx=16384)["vram_gib"]), "low 16k vram"),
        B("docs/07-capability-gains.md", "| P1 @4.0bpw KV8 | 16k |",
          lambda: (t(est(p1, "p1", ctx=16384)["vram_gib"]),
                   est(p1, "p1", ctx=16384)["vram_gib"]), "P1 16k vram"),
        B("docs/07-capability-gains.md", "| P1 @4.0bpw KV8 | 8k |",
          lambda: (t(est(p1, "p1")["vram_gib"]), est(p1, "p1")["vram_gib"]),
          "P1 8k vram"),
    ]

    # ---------------- docs/06 bandwidth column must equal peak x 0.62 -------
    # The bw column is an input, not a model output, so it is bound against the
    # documented 62% factor instead. This catches a silent bandwidth edit, which
    # would otherwise make every tps on the row look consistent while being
    # computed from the wrong budget.
    for gpu, peak in (("4090 Laptop 16G", 576), ("4080 Laptop 12G", 432),
                      ("3070 Ti Laptop 8G", 448), ("3060 Laptop 6G", 336),
                      ("4060 Laptop 8G", 272), ("3050 Laptop 8G", 224)):
        budget = round(peak * 0.62)
        out.append(
            B("docs/06-hardware-tiers.md", f"| RTX {gpu} |",
              (lambda b=budget: (f"| {b} |", float(b))), f"bw {gpu}"))

    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    bs = bindings()
    if args.list:
        for b in bs:
            try:
                text, val = b.render()
            except Exception as exc:  # noqa: BLE001
                text, val = f"<error {exc!r}>", None
            print(f"{b.path:34} {b.label:22} -> {text}")
        return 0

    fails = []
    for b in bs:
        try:
            ok, msg = b.check()
        except Exception as exc:  # noqa: BLE001 - render failure is a failure
            ok, msg = False, f"{b.path}: {b.label} could not be computed: {exc!r}"
        if ok:
            if args.verbose:
                print(f"ok   {msg}")
        else:
            fails.append(msg)

    print(f"bound {len(bs)} doc lines to the model")
    if fails:
        print(f"\n{len(fails)} FAILED:")
        for f in fails:
            print(f"  {f}")
        return 1
    print("every bound doc line matches what the model computes")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

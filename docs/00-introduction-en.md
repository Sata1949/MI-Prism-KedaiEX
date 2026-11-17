# Moss KedaiEX — English Summary

**Fit a 27B-class model into 8 GB VRAM + 16 GB RAM, and make it better than what
fits there by default.**

Full documentation is in Chinese under `docs/`. This file exists so the project
is legible to non-Chinese readers; the numbers are identical to the Chinese
docs because they come from the same script.

## The problem, measured

A 27B Dense model at 4.5 bpw with a 16-bit KV cache and 8k context needs
**19.02 GiB of VRAM** on the reference laptop. The machine has **7.00 GiB
usable**. The decode ceiling is **0.8 tok/s** because 95% of the per-token time
is spent pushing spilled weights over PCIe.

| config | params | weights | VRAM | host | tok/s ceiling | fits? |
|---|---:|---:|---:|---:|---:|---|
| 27B reference, 4.5bpw, KV16 | 27.01B | 14.39 GiB | 19.02 | 0.35 | 0.8 | no (+12.02) |
| **P1 target, 4.0bpw, KV8** | **10.15B** | **4.89 GiB** | **6.45** | 0.35 | **51.1** | **yes (+0.55)** |
| P1 depth-only (no FFN cut, full vocab) | 12.14B | 5.98 GiB | 7.55 | 0.35 | 13.0 | no (+0.55) |
| P1 fallback, 32 layers | 13.92B | 7.41 GiB | 9.16 | 0.35 | 4.2 | no (+2.16) |
| P2 width-cut (the "just use a smaller model" control) | 9.26B | 3.83 GiB | 7.08 | 0.35 | 39.7 | marginal |
| P3 MoE, 12 experts top-2 | 28.39B | 7.76 GiB | 4.82 | 6.54 | 9.4 | yes |

## The three findings that drive the design

**1. Quantization alone cannot solve this.** Even at 2.0 bpw body with a 4-bit
KV cache, the 27B still misses by 1.42 GiB. And a 2 bpw 27B is worse than a
4-bit 8B model.

**2. Depth pruning alone cannot solve this either.** Going from 62 to 24 layers
removes 58% of parameters but only 56% of VRAM, because every layer carries a
KV slice. At 24 layers the deficit is still 1.43 GiB.

**3. Three levers together work, and the last step is the one that crosses zero.**

| step | VRAM | headroom | saved |
|---|---:|---:|---:|
| 27B as-is @4.5bpw, KV16 | 19.02 | −12.02 | — |
| vocab 262144 → 131072 | 18.53 | −11.53 | 0.49 |
| depth 62 → 26 | 8.49 | −1.49 | 10.04 |
| FFN 21504 → 18432 | 7.82 | −0.82 | 0.67 |
| bpw 4.5 → 4.0 | 7.27 | −0.27 | 0.55 |
| KV 16 → 8 bit | **6.45** | **+0.55** | 0.82 |

Structural pruning alone still leaves an 0.82 GiB deficit. Phase 1 cannot be
"pruning only".

## Why the numbers can be trusted

Every constant in the model was measured on the reference box, not copied from
a spec sheet — because laptop spec sheets over-promise by ~2x:

| constant | spec sheet | measured | note |
|---|---:|---:|---|
| VRAM usable | 8 GiB | **7.00 GiB** | desktop compositor takes ~1 GiB |
| RAM visible | 16 GB | **15.19 GiB** | iGPU reservation |
| PCIe link | Gen4 x16 | **Gen4 x8** | physical board routing, not power saving |
| RAM bandwidth | 5600 MT/s dual | **4800 single, 19.3 GB/s** | only 1 DIMM installed |
| Single-thread RAM read | — | **6.0 GB/s** | 3.2x penalty vs 8 threads |

An early version of the model used 45 GB/s for host RAM (a plausible
dual-channel number) and predicted 30+ tok/s for the MoE design. The real
figure is 9.4. That 130% error is why `notes/decisions/0003` pins every
constant to a measured value with the measurement method written next to it.

## Repository layout

```
configs/       architecture descriptions + acceptance thresholds (JSON)
docs/          full design docs (Chinese): intro, budget frontier, roadmap,
               eval protocol, risks, hardware benchmarks
notes/         architecture decision records
tools/         probe_hardware.py  measure the machine
               budget_model.py    params/VRAM/speed model, zero deps
               eval_gate.py       CI gate, non-zero exit when a config misses
               audit_docs.py      re-derive all 261 numbers quoted in the docs
               selfcheck.py       everything that must stay green
tests/         44 unit tests incl. monotonicity, CLI contracts, gate edges
```

## Two stages, opposite optimization axes

Bandwidth sensitivity turned out to be inverted between phases:

| config | VRAM bw 280→400 | host bw 17→68 |
|---|---|---|
| P1 (fully resident) | 51.1 → 73.0 (+43%) | 51.1 → 51.1 (0%) |
| P3 (experts in RAM) | 9.4 → 9.6 (+2%) | 9.4 → 30.0 (+219%) |

P1's effort belongs entirely on the GPU side; P3's entirely on the CPU side and
on prefetch. Nothing carries over. See `notes/decisions/0004`.

## Verify it yourself

```bash
python tools/probe_hardware.py          # measure this machine
python tools/budget_model.py            # budget table
python tools/budget_model.py --ablation # per-lever marginal cost
python tools/budget_model.py --explain  # per-token time/byte breakdown
python tools/audit_docs.py               # every doc number must reproduce
python tools/eval_gate.py configs/*.json
python -I -m unittest discover -s tests -v
python -I tools/selfcheck.py            # all of the above
```

## What this is not

- Not a quantization script. 4-bit quantizing a 27B gives 14.39 GiB, still
  double the budget.
- Not "just use a smaller model". P1 and P2 are different experiments; P2 is
  the control that decides whether compression beats substitution.
- Not verified on real weights yet. Every tok/s figure is a **ceiling** derived
  from a bandwidth model, not a measurement. `docs/04-eval-protocol.md` defines
  the falsification conditions, including "if real speed is below 50% of the
  prediction, the model is wrong and must be recalibrated".

## Status

- Phase 1 (Dense pruning 27B → 10.15B): budget and gate **done and passing**;
  weight surgery and QAD refill **not started**.
- Phase 2 (structural changes, native 27B): **blocked** — no CUDA toolchain on
  the reference machine, and 2:4 sparse needs it.
- Phase 3 (Dense → MoE, native 27B): budget **worked out**, toolchain
  **blocked**, expert splitting and prefetch **not started**.

The two blockers that gate everything are disk space (15.5 GiB free; a 4.5 bpw
27B is 14.39 GiB) and the missing C/CUDA toolchain. Both are environment
problems, not model problems.

# KI8 1.5.0-rc3 CUDA patch

Applies to **KI8-1.5.0-rc3** only. It adds and enforces the W8A16 path:

- `ki8_decode_w32`: warp-integer Peak8 decode for 32-wide weight groups;
- `ki8_select`: fused radius candidate selection;
- `cuda_peak.require_a16()` + `info()["a16_decode"]`: hard contract, no silent fallback;
- `run.py check --require-a16`: packs a real `1.5.0-rc3` checkpoint (Peak8 weights, BF16 activations), installs it, and compares `auto` against `reference` bit-for-bit on `347×511` (tail) and `3072×9216` (real projection);
- `run.py bench-speed --profile peak8-a16`: measures W8A16 directly;
- installer gate: rc3 tree only, post-apply verification, rollback on failure.

```sh
git -C /path/to/KI8-1.5.0-rc3 apply --check patches/ki8-1.5.0-rc3-cuda.patch
git -C /path/to/KI8-1.5.0-rc3 apply         patches/ki8-1.5.0-rc3-cuda.patch
KI8_REQUIRE_A16=1 python /path/to/KI8-1.5.0-rc3/run.py check --device cuda:0 --require-a16
```

Verified on RTX 3070 Ti Laptop 8 GB / torch 2.9.1+cu128: 30 CPU tests pass, `a16_decode=native-w32`, both checkpoint shapes bit-identical, `wire_elements=216224`.

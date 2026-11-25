# KI8 1.5.0-rc3 KEX DiT — W8A16 (Peak8 / W32)

One checkpoint, split into 40 parts of 128 MiB (last part ~6.8 MiB) because GitHub
release assets are capped at 2 GiB.

| File | Format | Bytes | SHA256 |
|---|---|---:|---|
| `MI-DIT-Prism-KedaiEX1_ki8_15peak8.safetensors` | KI8 1.5.0-rc3 Peak8/W32, W8A16 | 5,241,596,371 | `7064ed51…e572` |

The KEX BF16 baseline is not distributed in this release; this is the shipped
KI8 1.5.0-rc3 weight.

## Colab / Linux download

私有仓库需要 `GITHUB_TOKEN`（repo 权限）；或 `gh auth login` 后用 `gh release download`。

```bash
BASE=https://github.com/Sata1949/MI-Prism-KedaiEX/releases/download/kex-weights-v1.5rc3
mkdir -p /content/kex && cd /content/kex

for i in $(seq 0 39); do
  n=$(printf "%03d" "$i")
  curl -L --fail --retry 5 -H "Authorization: token $GITHUB_TOKEN" -O "$BASE/MI-DIT-Prism-KedaiEX1_ki8_15peak8.safetensors.part$n"
done

cat MI-DIT-Prism-KedaiEX1_ki8_15peak8.safetensors.part* > MI-DIT-Prism-KedaiEX1_ki8_15peak8.safetensors
sha256sum MI-DIT-Prism-KedaiEX1_ki8_15peak8.safetensors
# 7064ed51dc966d4c01d7c0d1c16c0e242cc01217955040c116659164c4c3e572
```

`SHA256SUMS` in this release carries every part hash.

## Loading

Put the file in `ComfyUI/models/diffusion_models/`. The KI8 1.5.0-rc3 codec
loads it as **W8A16**: Peak8 weights decoded by `ki8_decode_w32`, BF16
activations. The CUDA patch for the KI8 runtime is
`patches/ki8-1.5.0-rc3-cuda.patch` in this repository; `UNETLoader` uses the
ComfyUI codec path.

## License

Derivative of MOSS-Image SFT, Apache-2.0. Keep `NOTICE` and the modification
notice (KEX tail + Peak8 quantization) with any redistribution. See
`LICENSING-LAYERS.md`.

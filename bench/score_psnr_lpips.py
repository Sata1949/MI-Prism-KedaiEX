"""PSNR + LPIPS between two sampled arms on identical noise/condition."""
import argparse, json, math
import numpy as np
import torch
import lpips

p = argparse.ArgumentParser()
p.add_argument("--ref", type=str, required=True, help="reference .npy image (float 0..1)")
p.add_argument("--test", type=str, required=True)
p.add_argument("--ref-latent", type=str, default="")
p.add_argument("--test-latent", type=str, default="")
p.add_argument("--out", type=str, required=True)
args = p.parse_args()

def _load(path):
    if str(path).lower().endswith((".png", ".jpg", ".jpeg")):
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return np.load(path)

ref = _load(args.ref); test = _load(args.test)
ri = (ref * 255).round().astype("uint8"); ti = (test * 255).round().astype("uint8")
err = ri.astype("f8") - ti.astype("f8")
mse = float((err ** 2).mean())
psnr = 10 * math.log10(255.0 ** 2 / mse) if mse > 0 else None

def to_tensor(x):
    return torch.from_numpy(x.astype("f4") / 127.5 - 1).permute(2, 0, 1)[None]

out = {"rgb_psnr_db": psnr, "rgb_mse": mse, "rgb_max_abs_255": float(np.abs(err).max()),
       "rgb_mae": float(np.abs(err).mean())}
for backbone in ("alex", "squeeze"):
    net = lpips.LPIPS(net=backbone, version="0.1", pretrained=True, pnet_rand=False,
                      eval_mode=True, verbose=False).eval()
    with torch.inference_mode():
        v = float(net(to_tensor(ri.astype("f4")), to_tensor(ti.astype("f4"))).flatten()[0])
    out[f"lpips_{backbone}"] = v
if args.ref_latent and args.test_latent:
    a = np.load(args.ref_latent); b = np.load(args.test_latent)
    lmse = float(((a.astype("f8") - b.astype("f8")) ** 2).mean())
    s = float(np.square(a.astype("f8")).mean())
    out["latent_psnr_db"] = 10 * math.log10(s / lmse) if lmse > 0 else None
    out["latent_corr"] = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
out["ref"] = args.ref; out["test"] = args.test
print(json.dumps(out, indent=1))
open(args.out, "w").write(json.dumps(out, indent=1))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read a GGUF file's header without loading the weights.

Why this exists
---------------
`configs/ref_27b.json` v1 described a 62-layer dense transformer with
hidden 5376 / GQA 32/16 / FFN 21504 / tied embeddings. Every one of those
fields was wrong: the actual base model is 65 layers (17 full attention +
48 linear attention), hidden 5120, MQA 24/4, head_dim 256, FFN 17408,
vocab 248320, untied. The error survived because nobody opened the file.

This tool is the fix: architecture facts must be read from the artifact that
contains them, not remembered. A GGUF v3 header is a plain length-prefixed KV
table followed by tensor descriptors, so the whole structure is readable in
milliseconds even for an 11 GiB file.

Usage
-----
    python tools/gguf_peek.py MODEL.gguf              # summary
    python tools/gguf_peek.py MODEL.gguf --arch       # hyperparameters only
    python tools/gguf_peek.py MODEL.gguf --tensors    # tensor composition
    python tools/gguf_peek.py MODEL.gguf --json       # machine-readable
    python tools/gguf_peek.py MODEL.gguf --license    # license + base model

Exit codes
----------
0  parsed
2  not a readable GGUF v3 file
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from collections import Counter, defaultdict

# GGUF metadata value type ids
T_UINT8, T_INT8, T_UINT16, T_INT16 = 0, 1, 2, 3
T_UINT32, T_INT32, T_FLOAT32, T_BOOL = 4, 5, 6, 7
T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 8, 9, 10, 11, 12

FMT = {T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h",
       T_UINT32: "<I", T_INT32: "<i", T_FLOAT32: "<f", T_BOOL: "<?",
       T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d"}

# ggml tensor type -> (name, bits per weight). Approximation for block quants
# is fine here: we report element counts, not byte sizes.
GGML_TYPES = {
    0: ("F32", 32.0), 1: ("F16", 16.0), 2: ("Q4_0", 4.5), 3: ("Q4_1", 5.0),
    6: ("Q5_0", 5.5), 7: ("Q5_1", 6.0), 8: ("Q8_0", 8.5), 9: ("Q8_1", 9.0),
    10: ("Q2_K", 2.5625), 11: ("Q3_K", 3.4375), 12: ("Q4_K", 4.5),
    13: ("Q5_K", 5.5), 14: ("Q6_K", 6.5625), 15: ("Q8_K", 8.5),
    16: ("IQ2_XXS", 2.0625), 17: ("IQ2_XS", 2.3125), 18: ("IQ3_XXS", 3.0625),
    19: ("IQ1_S", 1.5625), 20: ("IQ4_NL", 4.5), 21: ("IQ3_S", 3.4375),
    22: ("IQ2_S", 2.5), 23: ("IQ4_XS", 4.25), 24: ("I8", 8.0),
    25: ("I16", 16.0), 26: ("I32", 32.0), 27: ("I64", 64.0),
    28: ("F64", 64.0), 29: ("IQ1_M", 1.75), 30: ("BF16", 16.0),
}


class Reader:
    def __init__(self, fh) -> None:
        self.fh = fh

    def raw(self, n: int) -> bytes:
        b = self.fh.read(n)
        if len(b) != n:
            raise ValueError(f"unexpected EOF (wanted {n}, got {len(b)})")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self) -> str:
        n = self.u64()
        return self.raw(n).decode("utf-8", "replace")

    def value(self, vtype: int):
        if vtype == T_STRING:
            return self.string()
        if vtype == T_ARRAY:
            etype = self.u32()
            n = self.u64()
            # guard against absurd lengths in a corrupt file
            if n > 10_000_000:
                raise ValueError(f"implausible array length {n}")
            return [self.value(etype) for _ in range(n)]
        fmt = FMT.get(vtype)
        if fmt is None:
            raise ValueError(f"unknown metadata type {vtype}")
        return struct.unpack(fmt, self.raw(struct.calcsize(fmt)))[0]


def parse(path: str) -> dict:
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        r = Reader(fh)
        magic = r.raw(4)
        if magic != b"GGUF":
            raise ValueError(f"not a GGUF file (magic={magic!r})")
        version = r.u32()
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        n_tensors = r.u64()
        n_kv = r.u64()
        if n_kv > 1_000_000 or n_tensors > 10_000_000:
            raise ValueError(f"implausible counts: {n_kv} kv, {n_tensors} tensors")

        kv: dict = {}
        for _ in range(n_kv):
            key = r.string()
            vtype = r.u32()
            kv[key] = r.value(vtype)

        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            ndim = r.u32()
            if ndim > 8:
                raise ValueError(f"tensor {name}: implausible ndim {ndim}")
            dims = struct.unpack(f"<{ndim}Q", r.raw(8 * ndim))
            ggml = r.u32()
            offset = r.u64()
            n_elems = 1
            for d in dims:
                n_elems *= d
            tensors.append({"name": name, "dims": list(dims),
                            "ggml_type": ggml, "offset": offset,
                            "elements": n_elems})

    return {"path": path, "file_bytes": size, "version": version,
            "kv": kv, "tensors": tensors}


def arch_summary(kv: dict) -> dict:
    """Pull the hyperparameters out of whichever <arch>.* prefix is present."""
    arch = kv.get("general.architecture")
    if not arch:
        return {}
    out = {}
    for k, v in kv.items():
        if k.startswith(arch + "."):
            out[k[len(arch) + 1:]] = v
    return out


def tensor_composition(tensors: list) -> dict:
    """Group elements by role, and detect per-layer structure differences."""
    per_layer: dict[int, Counter] = defaultdict(Counter)
    glob = Counter()
    types = Counter()
    for t in tensors:
        types[GGML_TYPES.get(t["ggml_type"], (str(t["ggml_type"]), 0))[0]] += 1
        name = t["name"]
        parts = name.split(".")
        if parts[0] == "blk" and len(parts) >= 3:
            li = int(parts[1])
            rest = ".".join(parts[2:])
            if "ssm" in rest:
                kind = "ssm"
            elif "attn" in rest:
                kind = "attn"
            elif "ffn" in rest:
                kind = "ffn"
            else:
                kind = "norm"
            per_layer[li][kind] += t["elements"]
        else:
            glob[".".join(parts[:2]) if len(parts) >= 2 else name] += t["elements"]

    # which layers carry which kinds
    kind_layers = defaultdict(list)
    for li, c in per_layer.items():
        for kind in c:
            kind_layers[kind].append(li)

    # group layers by their exact kind->elements signature
    sig = Counter()
    for li, c in per_layer.items():
        sig[tuple(sorted(c.items()))] += 1

    return {
        "tensor_count": len(tensors),
        "total_elements": sum(t["elements"] for t in tensors),
        "quant_types": dict(types),
        "global": {k: v for k, v in glob.items()},
        "kind_layer_counts": {k: len(v) for k, v in kind_layers.items()},
        "kind_layer_examples": {k: v[:8] for k, v in kind_layers.items()},
        "distinct_layer_signatures": [
            {"layers": n, "parts": {k: v for k, v in s}} for s, n in sig.items()
        ],
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--arch", action="store_true")
    ap.add_argument("--tensors", action="store_true")
    ap.add_argument("--license", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if not os.path.exists(args.path):
        print(f"ERROR file not found: {args.path}", file=sys.stderr)
        return 2
    try:
        g = parse(args.path)
    except Exception as exc:  # noqa: BLE001 - surfaced, not swallowed
        print(f"ERROR {exc}", file=sys.stderr)
        return 2

    kv = g["kv"]
    arch = arch_summary(kv)
    comp = tensor_composition(g["tensors"])

    report = {
        "path": g["path"],
        "file_bytes": g["file_bytes"],
        "file_gib": round(g["file_bytes"] / 1024 ** 3, 2),
        "gguf_version": g["version"],
        "general": {k: v for k, v in kv.items() if k.startswith("general.")},
        "arch_name": kv.get("general.architecture"),
        "arch": arch,
        "composition": comp,
    }

    if args.json:
        json.dump(report, sys.stdout, ensure_ascii=False, indent=2,
                  default=str)
        sys.stdout.write("\n")
        return 0

    if args.license:
        print(f"file        : {g['path']}  ({report['file_gib']} GiB)")
        for k in ("general.license", "general.license.name",
                  "general.license.link", "general.name",
                  "general.basename", "general.size_label",
                  "general.quantized_by", "general.quantization_version",
                  "general.base_model.0.name",
                  "general.base_model.0.organization",
                  "general.base_model.0.repo_url"):
            if k in kv:
                print(f"  {k:34} = {kv[k]}")
        return 0

    if args.arch:
        print(f"architecture: {report['arch_name']}")
        for k in sorted(arch):
            print(f"  {k:42} = {arch[k]}")
        return 0

    if args.tensors:
        print(f"tensors: {comp['tensor_count']}   "
              f"elements: {comp['total_elements']/1e9:.3f} B")
        print("\nquant types:", comp["quant_types"])
        print("\nglobal tensors (elements):")
        for k, v in sorted(comp["global"].items(), key=lambda x: -x[1]):
            print(f"  {k:30} {v/1e6:9.2f} M")
        print("\nlayers per kind:")
        for k, v in comp["kind_layer_counts"].items():
            print(f"  {k:10} {v:3d} layers   e.g. {comp['kind_layer_examples'][k][:6]}")
        print("\ndistinct per-layer signatures:")
        for s in comp["distinct_layer_signatures"]:
            parts = ", ".join(f"{k}={v/1e6:.1f}M" for k, v in s["parts"].items())
            print(f"  {s['layers']:3d} layers: {parts}")
        return 0

    # default summary
    print(f"file        : {g['path']}  ({report['file_gib']} GiB)")
    print(f"gguf        : v{g['version']}")
    print(f"name        : {kv.get('general.name')}")
    print(f"architecture: {report['arch_name']}")
    print(f"license     : {kv.get('general.license')}")
    print(f"quantized_by: {kv.get('general.quantized_by')}")
    print(f"base_model  : {kv.get('general.base_model.0.repo_url')}")
    print(f"tensors     : {comp['tensor_count']}  "
          f"({comp['total_elements']/1e9:.2f} B elements)")
    print()
    key_arch = ("block_count", "embedding_length", "feed_forward_length",
                "attention.head_count", "attention.head_count_kv",
                "attention.key_length", "context_length",
                "full_attention_interval", "nextn_predict_layers")
    for k in key_arch:
        if k in arch:
            print(f"  {k:34} = {arch[k]}")
    print()
    print("layer kinds:", comp["kind_layer_counts"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

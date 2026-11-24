"""Minimal safetensors reader that does not use the crashing mmap bindings.

The safetensors build in this venv segfaults when two safe_open handles are live
(and after repeated open/close), so every access here is a plain seek+read with the
header re-parsed per call. Fine for the handful of tensors used in comparisons.
"""
import json
import struct

import torch

_DT = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
       "I64": torch.int64, "I32": torch.int32, "I16": torch.int16,
       "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    meta = hdr.pop("__metadata__", None)
    return hdr, meta


def keys(path):
    hdr, _ = header(path)
    return list(hdr.keys())


def read(path, name, dtype=None):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        info = hdr[name]
        start, end = info["data_offsets"]
        f.seek(8 + n + start)
        buf = f.read(end - start)
    dt = dtype or _DT[info["dtype"]]
    raw = bytearray(buf)
    if dt == torch.bfloat16:
        t = torch.frombuffer(raw, dtype=torch.int16).view(torch.bfloat16)
    else:
        t = torch.frombuffer(raw, dtype=dt)
    return t.reshape(info["shape"]).clone()

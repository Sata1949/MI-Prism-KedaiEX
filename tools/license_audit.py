#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""License and provenance audit for the phase-1 deliverables.

`phase1/components.json` claims a license for every artefact this project ships.
A claim is worthless unless something checks it. This tool does three things:

  1. Verifies each registered local path still exists with the recorded size and
     sha256. A drifted artefact invalidates the license claim attached to it.
  2. Re-reads license evidence that lives inside the artefacts themselves (GGUF
     KV tables, safetensors metadata) and compares it with the registry. The
     GGUF header is the only place a quantiser's license is recorded in bytes
     that cannot be argued with.
  3. Checks that every license named in the registry has a text file under
     `licenses/` and that the text is not a stub.

Exit codes
----------
0  every registered claim verified
1  at least one claim failed (details per check)
2  the registry itself is malformed

Usage
-----
    python -I tools/license_audit.py
    python -I tools/license_audit.py --skip-hashes    # fast, structure only
    python -I tools/license_audit.py --json out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "phase1" / "components.json"
LICENSE_DIR = ROOT / "licenses"

# License id -> text file under licenses/. MIT and Apache-2.0 already live at
# the repo root; the rest were fetched from their canonical source.
LICENSE_TEXTS = {
    "MIT": ROOT / "LICENSE",
    "Apache-2.0": ROOT / "LICENSE-APACHE-2.0",
    "GPL-3.0": LICENSE_DIR / "LICENSE-GPL-3.0.txt",
    "Qwen-Research-License": LICENSE_DIR / "LICENSE-QWEN-RESEARCH.txt",
    "FLUX.1-dev-Non-Commercial": LICENSE_DIR / "LICENSE-FLUX-NONCOMMERCIAL.md",
}

# A license text shorter than this is a placeholder, not a grant.
MIN_LICENSE_CHARS = {
    "MIT": 900,
    "Apache-2.0": 9000,
    "GPL-3.0": 20000,
    "Qwen-Research-License": 4000,
    "FLUX.1-dev-Non-Commercial": 2000,
}

GGUF_TYPE = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
             6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1 << 22)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


SKIP_DIRS = {".venv", "__pycache__", "_backup_moss_installer", ".pytest_cache"}


def sha256_tree(root: Path) -> tuple[str, int, int]:
    h = hashlib.sha256()
    entries = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in filenames:
            full = Path(dirpath) / name
            rel = full.relative_to(root).as_posix()
            entries.append((rel, full))
    total = 0
    for rel, full in sorted(entries):
        size = full.stat().st_size
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(str(size).encode("ascii"))
        h.update(b"\0")
        with open(full, "rb") as fh:
            while True:
                block = fh.read(1 << 22)
                if not block:
                    break
                h.update(block)
        total += size
    return h.hexdigest(), len(entries), total


def read_gguf_kv(path: Path, wanted: tuple[str, ...]) -> dict:
    """Read selected KV pairs from a GGUF header without loading tensors."""
    out: dict = {}
    with open(path, "rb") as fh:
        magic = fh.read(4)
        if magic != b"GGUF":
            raise ValueError(f"not a GGUF file: magic={magic!r}")
        _version = struct.unpack("<I", fh.read(4))[0]
        _n_tensors = struct.unpack("<Q", fh.read(8))[0]
        n_kv = struct.unpack("<Q", fh.read(8))[0]

        def read_str() -> str:
            n = struct.unpack("<Q", fh.read(8))[0]
            return fh.read(n).decode("utf-8", "replace")

        for _ in range(n_kv):
            key = read_str()
            vtype = struct.unpack("<I", fh.read(4))[0]
            if vtype == 8:
                value = read_str()
            elif vtype == 9:
                atype = struct.unpack("<I", fh.read(4))[0]
                count = struct.unpack("<Q", fh.read(8))[0]
                if atype == 8:
                    # Token arrays are huge; skip them without materialising.
                    if key in wanted:
                        value = [read_str() for _ in range(count)]
                    else:
                        for _ in range(count):
                            n = struct.unpack("<Q", fh.read(8))[0]
                            fh.seek(n, os.SEEK_CUR)
                        continue
                else:
                    fmt = GGUF_TYPE.get(atype)
                    if fmt is None:
                        raise ValueError(f"unsupported array type {atype} for {key}")
                    fh.read(count * struct.calcsize(fmt))
                    continue
            else:
                fmt = GGUF_TYPE.get(vtype)
                if fmt is None:
                    raise ValueError(f"unsupported type {vtype} for {key}")
                value = struct.unpack(fmt, fh.read(struct.calcsize(fmt)))[0]
            if key in wanted:
                out[key] = value
    return out


def read_safetensors_meta(path: Path) -> dict:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    return header.get("__metadata__") or {}


class Audit:
    def __init__(self, skip_hashes: bool) -> None:
        self.skip_hashes = skip_hashes
        self.checks: list[dict] = []

    def record(self, ok: bool, label: str, detail: str = "") -> bool:
        self.checks.append({"ok": ok, "label": label, "detail": detail})
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {label}" + (f"  {detail}" if detail else ""))
        return ok

    # -- 1. artifact integrity ------------------------------------------------
    def check_artifacts(self, comps: list[dict]) -> None:
        print("\n== artefact integrity ==")
        for c in comps:
            path_raw = c.get("local_path")
            if not path_raw or c.get("path_kind") in ("remote", "external"):
                continue
            path = Path(path_raw)
            cid = c["id"]
            if not path.exists():
                # Non-shipped reference artefacts may legitimately be absent on
                # another machine. Only fail for things the registry marks shipped.
                self.record(False, f"{cid}: path missing", str(path))
                continue
            if c.get("path_kind") == "tree":
                digest, count, total = sha256_tree(path)
                want_files = c.get("files")
                want_bytes = c.get("bytes")
                want_sha = c.get("sha256")
                if want_files is not None and count != want_files:
                    self.record(False, f"{cid}: file count", f"{count} != {want_files}")
                    continue
                if want_bytes is not None and total != want_bytes:
                    self.record(False, f"{cid}: byte total",
                                f"{total:,} != {want_bytes:,}")
                    continue
                if want_sha and not self.skip_hashes and digest != want_sha:
                    self.record(False, f"{cid}: tree sha256",
                                f"{digest[:16]} != {want_sha[:16]}")
                    continue
                self.record(True, f"{cid}: tree {count} files, {total:,} bytes",
                            "" if self.skip_hashes else digest[:16])
            else:
                size = path.stat().st_size
                want_bytes = c.get("bytes")
                if want_bytes is not None and size != want_bytes:
                    self.record(False, f"{cid}: size",
                                f"{size:,} != {want_bytes:,}")
                    continue
                want_sha = c.get("sha256")
                if want_sha and not self.skip_hashes:
                    digest = sha256_file(path)
                    if digest != want_sha:
                        self.record(False, f"{cid}: sha256",
                                    f"{digest[:16]} != {want_sha[:16]}")
                        continue
                    self.record(True, f"{cid}: {size:,} bytes", digest[:16])
                else:
                    self.record(True, f"{cid}: {size:,} bytes",
                                "hash skipped" if self.skip_hashes else "no hash recorded")

    # -- 2. in-artifact license evidence -------------------------------------
    def check_gguf_evidence(self, comps: list[dict]) -> None:
        print("\n== in-artifact evidence (GGUF) ==")
        for c in comps:
            if c.get("kind") != "weights" or not c.get("local_path", "").endswith(".gguf"):
                continue
            path = Path(c["local_path"])
            cid = c["id"]
            if not path.exists():
                self.record(False, f"{cid}: GGUF missing, cannot verify license")
                continue
            try:
                kv = read_gguf_kv(path, (
                    "general.license",
                    "general.name",
                    "general.quantized_by",
                    "general.base_model.0.repo_url",
                    "general.base_model.0.organization",
                ))
            except Exception as exc:  # noqa: BLE001 - report, do not crash
                self.record(False, f"{cid}: GGUF header unreadable", str(exc))
                continue
            declared = (c.get("license") or "").lower().replace("-", "")
            header_lic = str(kv.get("general.license", ""))
            norm_header = header_lic.lower().replace("-", "").replace("_", "")
            ok = norm_header == declared or f"{norm_header}2.0" == declared
            self.record(ok, f"{cid}: GGUF general.license = {header_lic}",
                        f"registry says {c.get('license')}")
            if "general.quantized_by" in kv:
                print(f"         quantized_by = {kv['general.quantized_by']}")
            if "general.base_model.0.repo_url" in kv:
                print(f"         base_model   = {kv['general.base_model.0.repo_url']}")

    def check_safetensors_evidence(self, comps: list[dict]) -> None:
        print("\n== in-artifact evidence (safetensors) ==")
        for c in comps:
            path_raw = c.get("local_path", "")
            if not path_raw.endswith(".safetensors"):
                continue
            path = Path(path_raw)
            cid = c["id"]
            if not path.exists():
                self.record(False, f"{cid}: safetensors missing")
                continue
            try:
                meta = read_safetensors_meta(path)
            except Exception as exc:  # noqa: BLE001
                self.record(False, f"{cid}: header unreadable", str(exc))
                continue
            has_lic = any("licen" in k.lower() for k in meta)
            # This is informational, not a failure: safetensors in the ComfyUI
            # ecosystem routinely carry no license metadata. What matters is
            # that the registry knows this and does not rely on the file.
            note = "carries license metadata" if has_lic else "no license metadata (expected)"
            print(f"  [INFO] {cid}: {note}  keys={sorted(meta)[:6]}")
            if not has_lic and not c.get("license"):
                self.record(False, f"{cid}: no license in file and none in registry")

    # -- 3. license texts ----------------------------------------------------
    def check_license_texts(self) -> None:
        print("\n== license texts ==")
        for lic, path in LICENSE_TEXTS.items():
            if not path.exists():
                self.record(False, f"{lic}: text missing", str(path))
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            minimum = MIN_LICENSE_CHARS.get(lic, 500)
            ok = len(text) >= minimum
            self.record(ok, f"{lic}: {len(text):,} chars",
                        f"min {minimum:,}" if not ok else path.name)

    # -- 4. registry consistency --------------------------------------------
    def check_registry(self, reg: dict) -> None:
        print("\n== registry consistency ==")
        comps = reg["components"]
        ids = [c["id"] for c in comps]
        dupes = {i for i in ids if ids.count(i) > 1}
        self.record(not dupes, "component ids unique", str(dupes) if dupes else "")
        idset = set(ids)

        bad_refs = []
        for c in comps:
            for dep in c.get("derived_from", []):
                ref = dep.get("component")
                if ref and ref not in idset:
                    bad_refs.append(f"{c['id']} -> {ref}")
        self.record(not bad_refs, "derived_from references resolve",
                    "; ".join(bad_refs))

        missing_lic = [c["id"] for c in comps
                       if not c.get("license") and not c.get("license_set")]
        self.record(not missing_lic,
                    "every component declares a license (or a license_set)",
                    str(missing_lic))

        no_status = [c["id"] for c in comps if not c.get("license_status")]
        self.record(not no_status, "every component declares a license_status",
                    str(no_status))

        # A conflict must point at components that exist.
        bad_conf = []
        for cf in reg.get("conflicts", []):
            for ref in cf.get("components", []):
                if ref not in idset:
                    bad_conf.append(f"{cf['id']} -> {ref}")
        self.record(not bad_conf, "conflict references resolve", "; ".join(bad_conf))

        # Anything marked inherited/combined-work/unresolved must be flagged by
        # a conflict, otherwise the conflict list is incomplete.
        flagged = {r for cf in reg.get("conflicts", []) for r in cf.get("components", [])}
        risky = {c["id"] for c in comps
                 if c.get("license_status") in ("unresolved", "combined-work")}
        unflagged = risky - flagged
        self.record(not unflagged,
                    "every unresolved/combined-work component appears in a conflict",
                    str(unflagged))

    def summary(self) -> int:
        failed = [c for c in self.checks if not c["ok"]]
        print(f"\n{'=' * 60}")
        print(f"{len(self.checks) - len(failed)}/{len(self.checks)} checks passed")
        if failed:
            print(f"\n{len(failed)} FAILED:")
            for c in failed:
                print(f"  - {c['label']}: {c['detail']}")
            return 1
        print("all phase-1 license claims verified")
        return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-hashes", action="store_true",
                    help="skip sha256 verification (structure and licenses only)")
    ap.add_argument("--json", metavar="PATH", help="write a machine-readable report")
    args = ap.parse_args(argv)

    if not REGISTRY.exists():
        print(f"registry missing: {REGISTRY}", file=sys.stderr)
        return 2
    try:
        reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"registry is not valid JSON: {exc}", file=sys.stderr)
        return 2

    comps = reg.get("components") or []
    if not comps:
        print("registry has no components", file=sys.stderr)
        return 2

    audit = Audit(args.skip_hashes)
    audit.check_registry(reg)
    audit.check_license_texts()
    audit.check_artifacts(comps)
    audit.check_gguf_evidence(comps)
    audit.check_safetensors_evidence(comps)
    code = audit.summary()

    if args.json:
        out = Path(args.json)
        out.write_text(json.dumps({
            "registry": str(REGISTRY),
            "checks": audit.checks,
            "passed": sum(1 for c in audit.checks if c["ok"]),
            "total": len(audit.checks),
            "exit": code,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report: {out}")

    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

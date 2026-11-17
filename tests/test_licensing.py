"""Tests for licensing artifacts and the GGUF header reader.

The licensing tests exist because a license file that looks right but is missing
a clause is worse than no license file: it reads as a grant it does not make.
The GGUF tests exist because the project's entire architecture description was
once wrong for months; reading the artifact is the only defence.

Run: python -I -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import gguf_peek  # noqa: E402

# A real model path on the development machine. Tests that need it skip when
# absent, so the suite stays runnable on a clean checkout.
LOCAL_GGUF = Path("D:/moss-image/moss27b/models/ms_dl/"
                  "Qwen3.8-27B-UD-IQ3_S.gguf")


def norm(text: str) -> str:
    """Collapse whitespace so line-wrapping inside clauses does not matter."""
    return re.sub(r"\s+", " ", text)


class TestMitLicense(unittest.TestCase):
    def setUp(self) -> None:
        self.text = norm((ROOT / "LICENSE").read_text(encoding="utf-8"))

    def test_title(self):
        self.assertIn("MIT License", self.text)

    def test_copyright_holder(self):
        self.assertRegex(self.text, r"Copyright \(c\) \d{4} .+")

    def test_grant_clause(self):
        self.assertIn("Permission is hereby granted, free of charge", self.text)
        self.assertIn("without restriction", self.text)

    def test_all_enumerated_rights(self):
        for right in ("use", "copy", "modify", "merge", "publish",
                      "distribute", "sublicense", "sell"):
            self.assertIn(right, self.text, f"missing right: {right}")

    def test_notice_preservation(self):
        self.assertIn("The above copyright notice and this permission notice "
                      "shall be included", self.text)

    def test_warranty_disclaimer(self):
        self.assertIn('THE SOFTWARE IS PROVIDED "AS IS"', self.text)
        self.assertIn("WARRANTIES OF MERCHANTABILITY", self.text)
        self.assertIn("FITNESS FOR A PARTICULAR PURPOSE", self.text)

    def test_liability_disclaimer(self):
        self.assertIn("IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE "
                      "LIABLE", self.text)
        self.assertIn("DAMAGES OR OTHER LIABILITY", self.text)


class TestApacheLicense(unittest.TestCase):
    def setUp(self) -> None:
        self.text = norm((ROOT / "LICENSE-APACHE-2.0").read_text(
            encoding="utf-8"))

    def test_identity(self):
        self.assertIn("Apache License", self.text)
        self.assertIn("Version 2.0, January 2004", self.text)

    def test_all_nine_sections_present(self):
        for n, title in enumerate([
                "Definitions", "Grant of Copyright License",
                "Grant of Patent License", "Redistribution",
                "Submission of Contributions", "Trademarks",
                "Disclaimer of Warranty", "Limitation of Liability",
                "Accepting Warranty or Additional Liability"], start=1):
            self.assertIn(f"{n}. {title}", self.text,
                          f"missing section {n}: {title}")

    def test_ends_with_terms_marker(self):
        self.assertIn("END OF TERMS AND CONDITIONS", self.text)

    def test_redistribution_obligations_named(self):
        # 4a/4b/4c are the clauses this project must actually satisfy.
        self.assertIn("You must give any other recipients", self.text)
        self.assertIn("cause any modified files to carry prominent notices",
                      self.text)
        self.assertIn("You must retain", self.text)

    def test_known_fingerprint_length(self):
        # The canonical text is ~202 lines; a truncated download would be far
        # shorter. Guards against a partial fetch being committed.
        lines = (ROOT / "LICENSE-APACHE-2.0").read_text(
            encoding="utf-8").splitlines()
        self.assertGreater(len(lines), 190)
        self.assertLess(len(lines), 260)


class TestNoticeFiles(unittest.TestCase):
    def setUp(self) -> None:
        self.text = (ROOT / "NOTICE").read_text(encoding="utf-8")

    def test_names_base_model_and_license(self):
        self.assertIn("Qwen3.8-27B", self.text)
        self.assertIn("Apache License 2.0", self.text)

    def test_declares_modifications(self):
        # Apache-2.0 4b: modified files must carry prominent notices.
        for action in ("pruning", "quantization", "vocabulary"):
            self.assertIn(action, self.text.lower())

    def test_trademark_disclaimer_present(self):
        self.assertIn("trademark", self.text.lower())
        self.assertIn("not endorsed by", self.text.lower())

    def test_points_at_both_license_texts(self):
        self.assertIn("LICENSE", self.text)
        self.assertIn("LICENSE-APACHE-2.0", self.text)


class TestShippedConfigsAreApacheLicensed(unittest.TestCase):
    def test_config_declares_provenance_and_upstream(self):
        cfg = json.loads((ROOT / "configs" / "qwen38_27b_real.json").read_text(
            encoding="utf-8"))
        self.assertEqual(cfg["role"], "baseline")
        prov = cfg["provenance"]
        self.assertIn("Qwen/Qwen3.8-27B", prov["base_model"])
        # provenance must say how it was obtained, not just that it was
        self.assertIn("GGUF", prov["extracted_by"])

    def test_real_arch_matches_gguf_expectations(self):
        cfg = json.loads((ROOT / "configs" / "qwen38_27b_real.json").read_text(
            encoding="utf-8"))
        a = cfg["arch"]
        self.assertEqual(a["hidden"], 5120)
        self.assertEqual(a["layers"], 65)
        self.assertEqual(a["full_attention_layers"], 17)
        self.assertEqual(a["linear_attention_layers"], 48)
        self.assertEqual(a["heads"], 24)
        self.assertEqual(a["kv_heads"], 4)
        self.assertEqual(a["head_dim"], 256)
        self.assertEqual(a["ffn"], 17408)
        self.assertEqual(a["vocab"], 248320)
        self.assertFalse(a["tie_embed"])


# --------------------------------------------------------------------------
# GGUF reader
# --------------------------------------------------------------------------
def make_gguf(path: Path, kv: dict, tensors: list) -> None:
    """Write a minimal GGUF v3 file with string/int/array metadata."""
    def s(x: str) -> bytes:
        b = x.encode("utf-8")
        return len(b).to_bytes(8, "little") + b

    def val(v):
        if isinstance(v, bool):
            return (7).to_bytes(4, "little") + (1 if v else 0).to_bytes(1, "little")
        if isinstance(v, int):
            return (4).to_bytes(4, "little") + v.to_bytes(4, "little")
        if isinstance(v, float):
            import struct as _s
            return (6).to_bytes(4, "little") + _s.pack("<f", v)
        if isinstance(v, str):
            return (8).to_bytes(4, "little") + s(v)
        if isinstance(v, list):
            out = (9).to_bytes(4, "little") + (8).to_bytes(4, "little")
            out += len(v).to_bytes(8, "little")
            for item in v:
                out += s(item)
            return out
        raise TypeError(type(v))

    buf = bytearray()
    buf += b"GGUF" + (3).to_bytes(4, "little")
    buf += len(tensors).to_bytes(8, "little")
    buf += len(kv).to_bytes(8, "little")
    for k, v in kv.items():
        buf += s(k) + val(v)
    for name, dims, ggml in tensors:
        buf += s(name) + len(dims).to_bytes(4, "little")
        for d in dims:
            buf += d.to_bytes(8, "little")
        buf += ggml.to_bytes(4, "little") + (0).to_bytes(8, "little")
    path.write_bytes(bytes(buf))


class TestGgufParse(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.gguf"
            make_gguf(p, {"general.architecture": "qwen35",
                          "general.license": "apache-2.0",
                          "qwen35.block_count": 65},
                      [("token_embd.weight", [16, 8], 1)])
            g = gguf_peek.parse(str(p))
        self.assertEqual(g["version"], 3)
        self.assertEqual(g["kv"]["general.license"], "apache-2.0")
        self.assertEqual(len(g["tensors"]), 1)
        self.assertEqual(g["tensors"][0]["elements"], 128)

    def test_rejects_bad_magic(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.bin"
            p.write_bytes(b"NOPE" + b"\x00" * 32)
            with self.assertRaises(ValueError):
                gguf_peek.parse(str(p))

    def test_rejects_truncated(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "trunc.gguf"
            p.write_bytes(b"GGUF" + (3).to_bytes(4, "little") + b"\x00" * 4)
            with self.assertRaises(ValueError):
                gguf_peek.parse(str(p))

    def test_arch_summary_filters_by_prefix(self):
        kv = {"general.architecture": "qwen35",
              "qwen35.block_count": 65,
              "other.key": 1}
        got = gguf_peek.arch_summary(kv)
        self.assertEqual(got, {"block_count": 65})

    def test_composition_groups_kinds(self):
        tensors = [
            {"name": "blk.0.attn_q.weight", "dims": [], "ggml_type": 1,
             "offset": 0, "elements": 100},
            {"name": "blk.0.ffn_up.weight", "dims": [], "ggml_type": 1,
             "offset": 0, "elements": 200},
            {"name": "blk.0.ssm_a", "dims": [], "ggml_type": 0,
             "offset": 0, "elements": 50},
            {"name": "blk.1.attn_q.weight", "dims": [], "ggml_type": 1,
             "offset": 0, "elements": 100},
            {"name": "token_embd.weight", "dims": [], "ggml_type": 1,
             "offset": 0, "elements": 999},
        ]
        comp = gguf_peek.tensor_composition(tensors)
        self.assertEqual(comp["kind_layer_counts"]["ssm"], 1)
        self.assertEqual(comp["kind_layer_counts"]["attn"], 2)
        self.assertEqual(comp["total_elements"], 1449)

    def test_cli_missing_file_exits_2(self):
        proc = subprocess.run(
            [sys.executable, "-I", str(ROOT / "tools" / "gguf_peek.py"),
             "no-such-file.gguf"],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 2)


@unittest.skipUnless(LOCAL_GGUF.exists(), "local Qwen3.8 GGUF not present")
class TestRealModelHeader(unittest.TestCase):
    """Guard the architecture facts the docs depend on, straight from the file."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.g = gguf_peek.parse(str(LOCAL_GGUF))
        cls.kv = cls.g["kv"]
        cls.arch = gguf_peek.arch_summary(cls.kv)

    def test_license_is_apache(self):
        self.assertEqual(self.kv.get("general.license"), "apache-2.0")

    def test_base_model_points_at_qwen(self):
        self.assertIn("Qwen/Qwen3.8-27B",
                      self.kv.get("general.base_model.0.repo_url", ""))

    def test_hyperparameters(self):
        a = self.arch
        self.assertEqual(a["block_count"], 65)
        self.assertEqual(a["embedding_length"], 5120)
        self.assertEqual(a["feed_forward_length"], 17408)
        self.assertEqual(a["attention.head_count"], 24)
        self.assertEqual(a["attention.head_count_kv"], 4)
        self.assertEqual(a["attention.key_length"], 256)
        self.assertEqual(a["context_length"], 262144)

    def test_hybrid_structure(self):
        comp = gguf_peek.tensor_composition(self.g["tensors"])
        counts = comp["kind_layer_counts"]
        self.assertEqual(counts["ssm"], 48, "linear-attention layer count")
        self.assertEqual(counts["attn"], 65)
        self.assertEqual(counts["ffn"], 65)

    def test_embeddings_are_untied(self):
        names = {t["name"] for t in self.g["tensors"]}
        self.assertIn("token_embd.weight", names)
        self.assertIn("output.weight", names)

    def test_parameter_count_matches_config(self):
        cfg = json.loads((ROOT / "configs" / "qwen38_27b_real.json").read_text(
            encoding="utf-8"))
        total_b = sum(t["elements"] for t in self.g["tensors"]) / 1e9
        # config declares 27.25 (hand sum); file sums to 27.32 including the
        # nextn block. Allow that gap explicitly rather than fudging the number.
        self.assertAlmostEqual(total_b, cfg["params_b"], delta=0.10)

    def test_file_is_still_too_big_for_7gib(self):
        gib = self.g["file_bytes"] / 1024 ** 3
        self.assertGreater(gib, 7.0,
                           "if this ever fits, revisit the whole phase-1 premise")

    @unittest.skipUnless(os.environ.get("MOSS_DEEP_TESTS"),
                         "set MOSS_DEEP_TESTS=1 to assert exact element counts")
    def test_exact_element_counts(self):
        cfg = json.loads((ROOT / "configs" / "qwen38_27b_real.json").read_text(
            encoding="utf-8"))
        me = cfg["measured_elements"]
        by_name = {t["name"]: t["elements"] for t in self.g["tensors"]}
        self.assertEqual(by_name["token_embd.weight"], me["token_embd"])
        self.assertEqual(by_name["output.weight"], me["output"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

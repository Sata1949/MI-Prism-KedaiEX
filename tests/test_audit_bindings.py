"""Tests for the doc-line binding audit.

The point of these tests is that the audit must FAIL when a doc drifts. A test
suite that only checks the happy path would reproduce exactly the defect the
binding audit was written to fix.

Run: python -I -m unittest discover -s tests -v
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import audit_bindings as ab  # noqa: E402


def run_audit() -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-I", str(ROOT / "tools" / "audit_bindings.py")],
        capture_output=True, text=True, timeout=180)
    return proc.returncode, proc.stdout + proc.stderr


class TestBindingsResolve(unittest.TestCase):
    def test_every_binding_renders_without_error(self):
        for b in ab.bindings():
            text, value = b.render()      # must not raise
            self.assertIsInstance(text, str, b.label)
            self.assertTrue(text, f"{b.label} rendered empty")
            self.assertIsNotNone(value, b.label)

    def test_every_binding_finds_its_line(self):
        for b in ab.bindings():
            f = ROOT / b.path
            self.assertTrue(f.exists(), f"{b.path} missing")
            text = f.read_text(encoding="utf-8")
            self.assertIn(b.anchor, text,
                          f"{b.label}: anchor {b.anchor!r} not in {b.path}")

    def test_no_duplicate_bindings(self):
        seen = set()
        for b in ab.bindings():
            key = (b.path, b.anchor, b.label)
            self.assertNotIn(key, seen, f"duplicate binding: {key}")
            seen.add(key)


class TestAuditPassesOnCleanTree(unittest.TestCase):
    def test_clean_audit_exits_zero(self):
        code, out = run_audit()
        self.assertEqual(code, 0, out[-2000:])


class TestAuditCatchesDrift(unittest.TestCase):
    """Each case mutates a real doc, runs the audit, then restores it."""

    def _mutate_and_check(self, path: str, old: str, new: str) -> None:
        p = ROOT / path
        original = p.read_text(encoding="utf-8")
        self.assertIn(old, original, f"mutation anchor missing in {path}")
        try:
            p.write_text(original.replace(old, new, 1), encoding="utf-8")
            code, out = run_audit()
            self.assertEqual(code, 1,
                             f"audit did not flag drift in {path}: {out[-600:]}")
        finally:
            p.write_text(original, encoding="utf-8")

    def test_headline_vram_number_drift(self):
        self._mutate_and_check(
            "README.md",
            "27B reference  19.02 GiB",
            "27B reference  29.02 GiB")

    def test_tier_throughput_drift(self):
        self._mutate_and_check(
            "README.md",
            "| 参考机（3070 Ti，278 GB/s） | 50.7 tok/s |",
            "| 参考机（3070 Ti，278 GB/s） | 59.7 tok/s |")

    def test_context_grid_drift(self):
        self._mutate_and_check(
            "docs/02-budget-frontier.md",
            "| 8k | 7.27 |",
            "| 8k | 9.27 |")

    def test_tier_bandwidth_column_drift(self):
        self._mutate_and_check(
            "docs/06-hardware-tiers.md",
            "| RTX 3050 Laptop 8G | 139 |",
            "| RTX 3050 Laptop 8G | 149 |")

    def test_context_table_drift(self):
        self._mutate_and_check(
            "docs/07-capability-gains.md",
            "| P1 @4.0bpw KV8 | 16k | 1.62 | 7.27",
            "| P1 @4.0bpw KV8 | 16k | 1.62 | 8.27")

    def test_deleted_line_is_caught(self):
        p = ROOT / "docs" / "06-hardware-tiers.md"
        original = p.read_text(encoding="utf-8")
        try:
            p.write_text(original.replace(
                "| RTX 4090 Laptop 16G | 357 | 65.2 | **86.3** | 46.8 | 9.6 |\n",
                "", 1), encoding="utf-8")
            code, out = run_audit()
            self.assertEqual(code, 1, out[-600:])
            self.assertIn("anchor not found", out)
        finally:
            p.write_text(original, encoding="utf-8")


class TestModelChangePropagates(unittest.TestCase):
    """If the model changes, the audit must demand a doc update.

    This runs in a COPY of the repo, in a subprocess. Mutating the real
    budget_model.py in-process is not safe: the module has already been
    imported, so a restore-on-disk still leaves the mutated value in memory and
    silently corrupts every later test and every later selfcheck step. That bug
    was observed, which is why this test is structured the way it is.
    """

    def test_changing_a_palette_breaks_the_audit(self):
        with tempfile.TemporaryDirectory() as td:
            work = Path(td) / "repo"
            shutil.copytree(
                ROOT, work,
                ignore=shutil.ignore_patterns(
                    ".git", "__pycache__", "*.pyc", "runs", "models"))

            model = work / "tools" / "budget_model.py"
            original = model.read_text(encoding="utf-8")
            mutated = original.replace(
                '"p1":  Palette(name="4.0bpw/KV8/8k [SHIPPED]", bpw_dense=4.0,\n'
                '                   bpw_expert=4.0, bpw_embed=6.0, kv_bits=8, '
                'ctx=8192),',
                '"p1":  Palette(name="4.0bpw/KV8/8k [SHIPPED]", bpw_dense=4.1,\n'
                '                   bpw_expert=4.1, bpw_embed=6.0, kv_bits=8, '
                'ctx=8192),', 1)
            self.assertNotEqual(mutated, original,
                                "palette mutation did not apply")
            model.write_text(mutated, encoding="utf-8")

            proc = subprocess.run(
                [sys.executable, "-I",
                 str(work / "tools" / "audit_bindings.py")],
                capture_output=True, text=True, timeout=180, cwd=str(work))
            self.assertEqual(
                proc.returncode, 1,
                "audit ignored a model change; docs would go stale silently. "
                f"Output: {(proc.stdout + proc.stderr)[-600:]}")


class TestCli(unittest.TestCase):
    def test_list_mode(self):
        proc = subprocess.run(
            [sys.executable, "-I", str(ROOT / "tools" / "audit_bindings.py"),
             "--list"],
            capture_output=True, text=True, timeout=180)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("->", proc.stdout)

    def test_verbose_mode(self):
        proc = subprocess.run(
            [sys.executable, "-I", str(ROOT / "tools" / "audit_bindings.py"),
             "--verbose"],
            capture_output=True, text=True, timeout=180)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ok   ", proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)

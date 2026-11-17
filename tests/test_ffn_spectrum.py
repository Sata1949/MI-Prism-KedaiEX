"""Tests for the FFN spectrum measurement tool.

Run: python -I -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import ffn_spectrum as fs  # noqa: E402

try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:  # noqa: BLE001
    HAVE_NUMPY = False


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
class TestEnergyCurve(unittest.TestCase):
    def test_uniform_spectrum_reaches_target_at_ratio(self):
        s = np.ones(100)
        self.assertEqual(fs.width_for_energy(s, 0.50), 50)
        self.assertEqual(fs.width_for_energy(s, 0.99), 99)

    def test_geometric_decay_needs_few(self):
        # ratio 0.9: energy is dominated by the first few
        s = np.array([0.9 ** i for i in range(200)])
        w = fs.width_for_energy(s, 0.99)
        self.assertLess(w, 50)

    def test_curve_is_monotone_and_ends_at_one(self):
        s = np.array([3.0, 2.0, 1.0, 0.5])
        c = fs.energy_curve(s)
        self.assertTrue(np.all(np.diff(c) >= 0))
        self.assertAlmostEqual(float(c[-1]), 1.0, places=12)

    def test_zero_spectrum_rejected(self):
        with self.assertRaises(ValueError):
            fs.energy_curve(np.zeros(10))

    def test_width_never_exceeds_rank(self):
        s = np.array([1.0, 1e-12, 1e-13])
        self.assertLessEqual(fs.width_for_energy(s, 0.999999999), len(s))


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
class TestMeasureLayer(unittest.TestCase):
    def test_low_rank_is_recovered(self):
        gate, up, down = fs.synthetic_layer(128, 512, rank=16, seed=3)
        m = fs.measure_layer(gate, up, down, 0.99)
        got = m["stacked"]["width_for_target"]
        self.assertLessEqual(abs(got - 16), 6, m["stacked"])

    def test_energy_at_857_is_high_for_low_rank(self):
        gate, up, down = fs.synthetic_layer(128, 512, rank=16, seed=4)
        m = fs.measure_layer(gate, up, down, 0.99)
        self.assertGreater(m["stacked"]["energy_at_857pct"], 0.99)

    def test_shape_errors_are_loud(self):
        rng = np.random.default_rng(0)
        g = rng.standard_normal((8, 16)).astype(np.float32)
        u = rng.standard_normal((8, 32)).astype(np.float32)   # wrong width
        d = rng.standard_normal((16, 8)).astype(np.float32)
        with self.assertRaises(ValueError):
            fs.measure_layer(g, u, d, 0.99)

    def test_down_transpose_error_is_loud(self):
        rng = np.random.default_rng(1)
        g = rng.standard_normal((8, 16)).astype(np.float32)
        u = rng.standard_normal((8, 16)).astype(np.float32)
        d = rng.standard_normal((8, 16)).astype(np.float32)   # not transposed
        with self.assertRaises(ValueError):
            fs.measure_layer(g, u, d, 0.99)

    def test_reports_both_views(self):
        gate, up, down = fs.synthetic_layer(64, 256, rank=8, seed=5)
        m = fs.measure_layer(gate, up, down, 0.99)
        self.assertIn("stacked", m)
        self.assertIn("composed", m)
        for v in ("stacked", "composed"):
            self.assertIn("width_for_target", m[v])
            self.assertIn("retain_pct", m[v])

    def test_retain_pct_matches_width(self):
        gate, up, down = fs.synthetic_layer(64, 256, rank=8, seed=6)
        m = fs.measure_layer(gate, up, down, 0.99)
        width = m["stacked"]["width_for_target"]
        rank = m["stacked"]["rank"]
        self.assertAlmostEqual(m["stacked"]["retain_pct"],
                               100.0 * width / rank, places=2)


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
class TestProbe(unittest.TestCase):
    def test_probe_passes_on_known_ranks(self):
        # Rank must stay clear of the matrix's own rank ceiling: the stacked
        # matrix is (2h, f), so its rank is at most min(2h, f) = 2h here.
        rows = fs.spectrum_probe(h=64, f=256, ranks=(8, 32, 64, 128))
        for r in rows:
            self.assertTrue(r["ok"], r)

    def test_full_rank_marked_not_recovered(self):
        # ceiling for h=64, f=256 is min(2h, f) = 128. Use it exactly.
        rows = fs.spectrum_probe(h=64, f=256, ranks=(128,))
        self.assertFalse(rows[0]["expect_recovered"])
        self.assertTrue(rows[0]["ok"], rows[0])

    def test_rank_above_ceiling_is_rejected(self):
        # Asking for more rank than the matrix can hold is a test bug, and the
        # tool must say so rather than return a meaningless number.
        with self.assertRaises(ValueError):
            fs.spectrum_probe(h=64, f=256, ranks=(256,))

    def test_json_roundtrip(self):
        # Default dims are small on purpose; --full is the minutes-long opt-in.
        out = subprocess.run(
            [sys.executable, "-I", str(ROOT / "tools" / "ffn_spectrum.py"),
             "--synthetic", "--json"],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        rows = json.loads(out.stdout)
        self.assertEqual(len(rows), 4)
        self.assertIn("true_rank", rows[0])
        self.assertIn("ok", rows[0])


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
class TestNpzPath(unittest.TestCase):
    def test_npz_roundtrip(self):
        rng = np.random.default_rng(7)
        gate = rng.standard_normal((32, 64)).astype(np.float32)
        up = rng.standard_normal((32, 64)).astype(np.float32)
        down = rng.standard_normal((64, 32)).astype(np.float32)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "layer.npz"
            np.savez(p, gate=gate, up=up, down=down)
            out = subprocess.run(
                [sys.executable, "-I",
                 str(ROOT / "tools" / "ffn_spectrum.py"),
                 "--npz", str(p), "--json"],
                capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr)
        m = json.loads(out.stdout)
        self.assertEqual(m["hidden"], 32)
        self.assertEqual(m["ffn"], 64)

    def test_missing_array_exits_2(self):
        rng = np.random.default_rng(8)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "partial.npz"
            np.savez(p, gate=rng.standard_normal((8, 16)).astype(np.float32))
            out = subprocess.run(
                [sys.executable, "-I",
                 str(ROOT / "tools" / "ffn_spectrum.py"), "--npz", str(p)],
                capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 2, out.stdout)

    def test_missing_file_exits_2(self):
        out = subprocess.run(
            [sys.executable, "-I", str(ROOT / "tools" / "ffn_spectrum.py"),
             "--npz", "does-not-exist.npz"],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 2)

    def test_transposed_tensors_are_accepted(self):
        """A checkpoint storing (f, h) layouts must not be silently misread."""
        rng = np.random.default_rng(9)
        gate = rng.standard_normal((64, 32)).astype(np.float32)  # (f, h)
        up = rng.standard_normal((64, 32)).astype(np.float32)
        down = rng.standard_normal((32, 64)).astype(np.float32)  # (h, f)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.npz"
            np.savez(p, gate=gate, up=up, down=down)
            out = subprocess.run(
                [sys.executable, "-I",
                 str(ROOT / "tools" / "ffn_spectrum.py"),
                 "--npz", str(p), "--json"],
                capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr)
        m = json.loads(out.stdout)
        self.assertEqual(m["hidden"], 32)
        self.assertEqual(m["ffn"], 64)

    def test_no_args_is_an_error(self):
        out = subprocess.run(
            [sys.executable, "-I", str(ROOT / "tools" / "ffn_spectrum.py")],
            capture_output=True, text=True, timeout=60)
        self.assertNotEqual(out.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

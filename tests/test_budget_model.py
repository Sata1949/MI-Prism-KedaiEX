"""Tests for the Moss KedaiEX budget model and deployment gate.

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

from budget_model import (  # noqa: E402
    Arch, Palette, estimate, p1, p2, p3, ref_27b, GIB,
)
import eval_gate  # noqa: E402


def tiny(**kw) -> Arch:
    base = dict(name="tiny", hidden=256, layers=2, heads=4, kv_heads=2,
                head_dim=64, ffn=512, vocab=1024, tie_embed=True)
    base.update(kw)
    return Arch(**base)


class TestParamCounting(unittest.TestCase):
    def test_single_layer_attention_math(self):
        a = tiny()
        h, kv, d, heads = a.hidden, a.kv_heads, a.head_dim, a.heads
        expected = h * heads * d + 2 * h * kv * d + heads * d * h
        self.assertEqual(a.attn_params_per_layer(), expected)

    def test_tied_embedding_counted_once(self):
        tied = tiny(tie_embed=True)
        untied = tiny(tie_embed=False)
        self.assertEqual(untied.params() - tied.params(), tiny().vocab * 256)

    def test_dense_mlp_has_three_projections(self):
        a = tiny(ffn=512)
        self.assertEqual(a.dense_mlp_params(), 2 * 3 * 256 * 512)

    def test_mla_has_no_separate_kv_cache_projection(self):
        a = tiny(kv_lora_rank=32)
        self.assertEqual(a.kv_per_layer_elems(), 2 * 32)
        self.assertGreater(a.mla_params_per_layer(), 0)

    def test_moe_total_exceeds_active(self):
        a = tiny(experts=8, experts_per_tok=2, expert_ffn=128, shared_experts=1)
        self.assertGreater(a.params(), a.active_params())
        # active = dense body + routed experts + shared
        self.assertEqual(a.expert_mlp_params(),
                         2 * 8 * 3 * 256 * 128)


class TestKv(unittest.TestCase):
    def test_kv_scales_linearly_with_ctx(self):
        a = tiny()
        self.assertAlmostEqual(a.kv_gib(2048) / a.kv_gib(1024), 2.0, places=6)

    def test_kv_halves_at_8_bit(self):
        a = tiny()
        self.assertAlmostEqual(a.kv_gib(1024, kv_bits=8) * 2,
                               a.kv_gib(1024, kv_bits=16), places=6)

    def test_windowed_decode_reads_less_than_full_kv(self):
        a = tiny(layers=8, global_every=4, window=64)
        self.assertLessEqual(a.decode_kv_gib(4096), a.kv_gib(4096))

    def test_full_attention_reads_all_kv(self):
        a = tiny(global_every=0)
        self.assertAlmostEqual(a.decode_kv_gib(4096), a.kv_gib(4096), places=9)


class TestEstimateShape(unittest.TestCase):
    def test_result_keys_stable(self):
        r = estimate(p1(), Palette())
        for key in ("arch", "palette", "params_total_b", "params_active_b",
                    "weights_gib", "kv_gib", "kv_decode_gib", "ctx", "vram_gib",
                    "spill_gib", "host_gib", "host_fit", "placement",
                    "decode_tok_s_ceiling"):
            self.assertIn(key, r)

    def test_effective_bpw_close_to_requested(self):
        # embedding at 6bpw pulls the average up from the 4.0 body setting
        r = estimate(p1(), Palette(bpw_dense=4.0, bpw_expert=4.0,
                                   bpw_embed=6.0))
        self.assertGreater(r["weights_gib"]["effective_bpw"], 3.9)
        self.assertLess(r["weights_gib"]["effective_bpw"], 4.8)

    def test_weights_scale_with_bpw(self):
        # Only the body scales; embeddings stay at 6 bpw, so the ratio is
        # below 2.0 by exactly the embedding share. Assert the exact relation
        # rather than a hand-waved "about double".
        a = p1()
        pal_lo = Palette(bpw_dense=4.0, bpw_expert=4.0, bpw_embed=6.0)
        pal_hi = Palette(bpw_dense=8.0, bpw_expert=8.0, bpw_embed=6.0)
        lo = estimate(a, pal_lo)["weights_gib"]
        hi = estimate(a, pal_hi)["weights_gib"]
        expected = (lo["dense"] * 2 + lo["embed"]) / (lo["dense"] + lo["embed"])
        self.assertAlmostEqual(hi["total"] / lo["total"], expected, places=3)

    def test_more_context_costs_vram(self):
        a = p1()
        small = estimate(a, Palette(ctx=4096, kv_bits=8))
        large = estimate(a, Palette(ctx=32768, kv_bits=8))
        self.assertGreater(large["vram_gib"], small["vram_gib"])

    def test_moe_puts_experts_on_host(self):
        r = estimate(p3(), Palette(bpw_dense=3.0, bpw_expert=2.2,
                                   bpw_embed=4.5))
        self.assertGreater(r["host_gib"], 4.0)
        self.assertLess(r["vram_gib"], 6.0)
        self.assertTrue(r["host_fit"])

    def test_27b_does_not_fit(self):
        r = estimate(ref_27b(), Palette(bpw_dense=4.5, bpw_expert=4.5,
                                        bpw_embed=6.0, ctx=8192, kv_bits=16))
        self.assertGreater(r["spill_gib"], 1.0)
        self.assertNotEqual(r["placement"], "resident")

    def test_placement_strings(self):
        resident = estimate(p2(), Palette(bpw_dense=3.0, bpw_expert=3.0,
                                          bpw_embed=6.0, ctx=8192, kv_bits=8,
                                          vram_gib=7.0))
        self.assertEqual(resident["placement"], "resident")
        self.assertEqual(resident["spill_gib"], 0.0)

    def test_over_budget_branch(self):
        r = estimate(p1(), Palette(bpw_dense=5.0, bpw_expert=5.0,
                                   bpw_embed=6.0, ctx=8192, kv_bits=8,
                                   vram_gib=7.0))
        self.assertGreater(r["spill_gib"], 0.0)
        self.assertTrue(r["placement"].startswith("partial")
                        or r["placement"].startswith("over"))


class TestMonotonicity(unittest.TestCase):
    """Every lever must move the number in the direction the docs claim."""

    def _vram(self, arch: Arch, pal: Palette) -> float:
        return estimate(arch, pal)["vram_gib"]

    def test_deeper_costs_more(self):
        base = dict(hidden=512, heads=8, kv_heads=4, head_dim=64, ffn=1024,
                    vocab=2048)
        shallow = Arch(name="s", layers=4, **base)
        deep = Arch(name="d", layers=8, **base)
        pal = Palette(ctx=2048, kv_bits=8)
        self.assertGreater(self._vram(deep, pal), self._vram(shallow, pal))

    def test_wider_ffn_costs_more(self):
        base = dict(hidden=512, layers=4, heads=8, kv_heads=4, head_dim=64,
                    vocab=2048)
        narrow = Arch(name="n", ffn=1024, **base)
        wide = Arch(name="w", ffn=4096, **base)
        pal = Palette(ctx=2048, kv_bits=8)
        self.assertGreater(self._vram(wide, pal), self._vram(narrow, pal))

    def test_bigger_vocab_costs_more(self):
        base = dict(hidden=512, layers=4, heads=8, kv_heads=4, head_dim=64,
                    ffn=1024)
        small = Arch(name="s", vocab=2048, **base)
        big = Arch(name="b", vocab=32768, **base)
        pal = Palette(ctx=2048, kv_bits=8)
        self.assertGreater(self._vram(big, pal), self._vram(small, pal))


class TestGate(unittest.TestCase):
    def test_p1_passes_default_gate(self):
        status, msgs = eval_gate.check(p1(), Palette(
            bpw_dense=4.0, bpw_expert=4.0, bpw_embed=6.0, kv_bits=8, ctx=8192),
            eval_gate.DEFAULT_GATE)
        self.assertEqual(status, 0, msgs)

    def test_oversized_config_fails(self):
        status, msgs = eval_gate.check(ref_27b(), Palette(
            bpw_dense=4.5, bpw_expert=4.5, bpw_embed=6.0, ctx=8192,
            kv_bits=16), eval_gate.DEFAULT_GATE)
        self.assertEqual(status, 2)
        self.assertTrue(any("FAIL" in m for m in msgs))

    def test_baseline_role_never_fails(self):
        status, msgs = eval_gate.check(ref_27b(), Palette(
            bpw_dense=16, bpw_expert=16, bpw_embed=16, ctx=8192, kv_bits=16),
            eval_gate.DEFAULT_GATE, role="baseline")
        self.assertEqual(status, 0)
        self.assertTrue(any("INFO" in m for m in msgs))

    def test_tight_headroom_warns(self):
        # p1 with 4.5bpw sits at exactly 7.00 GiB: 0 headroom => WARN, not FAIL
        pal = Palette(bpw_dense=4.5, bpw_expert=4.5, bpw_embed=6.0,
                      kv_bits=8, ctx=8192, vram_gib=7.0)
        gate = dict(eval_gate.DEFAULT_GATE)
        gate["headroom_gib_min"] = 0.25
        status, msgs = eval_gate.check(p1(), pal, gate)
        self.assertEqual(status, 1, msgs)

    def test_low_decode_fails(self):
        gate = dict(eval_gate.DEFAULT_GATE)
        gate["decode_tok_s_floor"] = 9999.0
        status, _ = eval_gate.check(p1(), Palette(bpw_dense=4.0), gate)
        self.assertEqual(status, 2)

    def test_bad_config_exits_3(self):
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "bad.json"
            bad.write_text('{"not_arch": 1}', encoding="utf-8")
            code = eval_gate.main([str(bad)])
            self.assertEqual(code, 3)

    def test_shipped_configs_are_green(self):
        code = eval_gate.main([str(ROOT / "configs" / "p1_target.json"),
                               str(ROOT / "configs" / "ref_27b.json")])
        self.assertEqual(code, 0)


class TestCli(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-I", str(ROOT / "tools" / "budget_model.py"),
             *args],
            capture_output=True, text=True, timeout=120)

    def test_json_is_parseable(self):
        out = self._run("--json")
        self.assertEqual(out.returncode, 0, out.stderr)
        rows = json.loads(out.stdout)
        self.assertGreaterEqual(len(rows), 4)
        for row in rows:
            self.assertIn("params_total_b", row)

    def test_arch_filter_returns_one_row(self):
        out = self._run("--arch", "p2", "--json")
        rows = json.loads(out.stdout)
        self.assertEqual(len(rows), 1)
        self.assertIn("P2", rows[0]["arch"])

    def test_set_override_takes_effect(self):
        base = json.loads(self._run("--arch", "p1", "--json").stdout)[0]
        big = json.loads(self._run("--arch", "p1", "--set", "ctx=32768",
                                   "--json").stdout)[0]
        self.assertGreater(big["vram_gib"], base["vram_gib"])

    def test_unknown_set_field_is_rejected(self):
        out = self._run("--set", "nonsense=1")
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("unknown palette field", out.stderr)

    def test_human_table_runs(self):
        out = self._run()
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("placement", out.stdout)

    def test_ablation_runs(self):
        out = self._run("--ablation")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("baseline P1", out.stdout)

    def test_explain_runs(self):
        out = self._run("--explain")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("per token", out.stdout)

    def test_explain_json_has_time_breakdown(self):
        out = self._run("--explain", "--json")
        rows = json.loads(out.stdout)
        for row in rows:
            m = row["ms_per_token"]
            parts = m["vram_resident"] + m["pcie_spill"] + m["host_stream"]
            self.assertAlmostEqual(parts, m["total"], delta=0.5)
            # ceiling must agree with the total it decomposes to
            self.assertAlmostEqual(1000.0 / m["total"],
                                   row["decode_tok_s_ceiling"], delta=0.5)

    def test_ablation_json_has_levers(self):
        out = self._run("--ablation", "--json")
        rows = json.loads(out.stdout)
        self.assertEqual(len(rows), 17)
        self.assertIn("lever", rows[0])
        self.assertIn("headroom_gib", rows[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)

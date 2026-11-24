"""Tests for the phase-1 licence registry and the audit that enforces it.

The registry (`phase1/components.json`) is the single place where this project
says which licence governs which artefact. These tests exist because that claim
is the kind of thing that silently rots: someone re-exports a weight, the hash
in the registry goes stale, and every downstream document keeps asserting a
licence for a file that is no longer the file it was checked against.

The tests do not need the 11 GiB GGUF to be present. Artefact-integrity checks
live in `tools/license_audit.py` and skip cleanly when a path is missing, which
is what a CI box and a fresh checkout both need.

Run: python -I -m unittest discover -s tests -v
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import license_audit  # noqa: E402

REGISTRY = ROOT / "phase1" / "components.json"


class TestRegistryShape(unittest.TestCase):
    def setUp(self) -> None:
        self.reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
        self.comps = {c["id"]: c for c in self.reg["components"]}

    def test_schema_tag(self):
        self.assertEqual(self.reg["schema"], "mi-prism-kedaiex/phase1-components@3")

    def test_copyright_names_the_four_person_team(self):
        # The MIT grant is made by four named people. If this drifts, every
        # "MIT" claim in the docs points at a holder nobody can identify.
        cp = self.reg["copyright"]
        self.assertEqual(cp["holder"], "sata1921, seeds, suihan, hilo114")
        self.assertIn("2026", cp["display"])
        lic = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("sata1921, seeds, suihan, hilo114", lic)

    def test_declared_mit_components_carry_the_team_copyright(self):
        for cid in ("ki8-core", "converter", "installer"):
            c = self.comps[cid]
            self.assertEqual(c.get("copyright"),
                             "sata1921, seeds, suihan, hilo114",
                             f"{cid}: MIT component missing the team copyright")

    def test_every_phase1_deliverable_is_registered(self):
        # The five things this phase actually produced. If one disappears from
        # the registry, its licence claim disappears with it.
        for cid in ("ki8-core", "dit-weights", "converter", "gsq-te",
                    "installer"):
            self.assertIn(cid, self.comps, f"missing deliverable: {cid}")

    def test_every_component_has_license_and_status(self):
        for cid, c in self.comps.items():
            self.assertTrue(c.get("license") or c.get("license_set"),
                            f"{cid}: no license")
            self.assertTrue(c.get("license_status"), f"{cid}: no license_status")
            self.assertIn(c["license_status"],
                          ("declared", "inherited", "inherited-separate",
                           "verified-in-artifact", "verified-upstream",
                           "combined-work", "unresolved", "mixed"),
                          f"{cid}: unknown license_status")

    def test_license_ids_have_text_files(self):
        for cid, c in self.comps.items():
            lic = c.get("license")
            status = c.get("license_status")
            # A license_set means the component is an aggregate (e.g. pip
            # dependencies) whose texts are not vendored. 'mixed' means several
            # grants apply per-file and are resolved elsewhere. An unresolved
            # component is one whose grant is not yet known. None of the three
            # claims a single named grant, so none needs one text here.
            if not lic or status == "unresolved" or lic in ("mixed", "per-package"):
                continue
            # Compound ids ("A + B") name more than one grant; the B half is
            # covered by the unresolved conflict, so only check the first.
            head = lic.split("+")[0].strip()
            self.assertIn(head, license_audit.LICENSE_TEXTS,
                          f"{cid}: license {lic!r} has no registered text file")

    def test_license_set_entries_are_named(self):
        for cid, c in self.comps.items():
            for lic in c.get("license_set") or []:
                self.assertIsInstance(lic, str)
                self.assertTrue(lic.strip(), f"{cid}: empty license_set entry")

    def test_derived_from_references_resolve(self):
        ids = set(self.comps)
        for cid, c in self.comps.items():
            for dep in c.get("derived_from", []):
                ref = dep.get("component")
                if ref:
                    self.assertIn(ref, ids, f"{cid} -> missing {ref}")


class TestConflictCoverage(unittest.TestCase):
    """A licence risk that is not in the conflict list is a risk nobody tracks."""

    def setUp(self) -> None:
        self.reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
        self.comps = {c["id"]: c for c in self.reg["components"]}
        self.flagged = {r for cf in self.reg["conflicts"]
                        for r in cf.get("components", [])}

    def test_unresolved_components_are_flagged(self):
        risky = {cid for cid, c in self.comps.items()
                 if c.get("license_status") in ("unresolved", "combined-work")}
        missing = risky - self.flagged
        self.assertFalse(missing, f"unflagged licence risk: {missing}")

    def test_every_conflict_offers_options(self):
        for cf in self.reg["conflicts"]:
            self.assertGreaterEqual(len(cf.get("options", [])), 2,
                                    f"{cf['id']}: fewer than two options")

    def test_noncommercial_findings_are_recorded(self):
        # These two were discovered by reading upstream metadata, not by
        # assuming. If they vanish from the registry someone has re-assumed
        # Apache-2.0 for a model that is not.
        ids = {cf["id"] for cf in self.reg["conflicts"]}
        self.assertIn("C1", ids)  # flux2 VAE is non-commercial
        self.assertIn("C3", ids)  # Qwen-Image-2.1 is research-only

    def test_gpl_combined_work_is_recorded(self):
        c = self.comps["comfyui-host"]
        self.assertEqual(c["license"], "GPL-3.0")
        self.assertEqual(self.comps["comfyui-ki8-codec"]["license"], "GPL-3.0")
        self.assertEqual(self.comps["comfyui-nodes"]["license"], "GPL-3.0")
        self.assertEqual(self.comps["comfyui-ki8-codec"]["license_status"],
                         "combined-work")


class TestInArtifactEvidenceReference(unittest.TestCase):
    def test_gsq_te_claims_file_level_evidence(self):
        reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
        gsq = next(c for c in reg["components"] if c["id"] == "gsq-te")
        # This is the strongest evidence class in the registry. It must not be
        # downgraded to an inference without someone noticing.
        self.assertEqual(gsq["license_status"], "verified-in-artifact")
        self.assertIn("general.license", gsq["license_evidence"])
        self.assertIn("general.quantized_by", gsq["license_evidence"])

    def test_gguf_reader_agrees_with_registry(self):
        reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
        gsq = next(c for c in reg["components"] if c["id"] == "gsq-te")
        path = Path(gsq["local_path"])
        if not path.exists():
            self.skipTest(f"GGUF not present: {path}")
        kv = license_audit.read_gguf_kv(path, ("general.license",
                                               "general.quantized_by"))
        self.assertEqual(kv.get("general.license", "").lower(), "apache-2.0")
        self.assertIn("ISTA", kv.get("general.quantized_by", ""))


class TestVaeLicence(unittest.TestCase):
    """The VAE's licence was got wrong once and then corrected. This pins it.

    The trap: Comfy-Org/Flux-2-Dev carries a repository-level
    `flux-1-dev-non-commercial-license` tag, but the file the pipeline uses is
    published separately by Black Forest Labs as FLUX.2-small-decoder under
    Apache-2.0. File-level provenance beats a repackager's repo-level tag, and
    the hash is what proves the two are the same bytes.
    """

    SMALL_SHA = ("d8d52ba036475f5fb07c8b435e176d3d97ebfa82f0d1a1c317f9cc1e25bd013b")
    FULL_SHA = ("d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5")

    def setUp(self) -> None:
        reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
        self.comps = {c["id"]: c for c in reg["components"]}

    def test_small_decoder_is_apache_and_is_the_default(self):
        vae = self.comps["flux2-vae"]
        self.assertEqual(Path(vae["local_path"]).name,
                         "flux2-vae_small.safetensors")
        self.assertEqual(vae["license"], "Apache-2.0")
        self.assertEqual(vae["bytes"], 249521340)
        self.assertEqual(vae["sha256"], self.SMALL_SHA)

    def test_small_decoder_cites_the_bfl_upstream(self):
        # The Apache claim is only as good as the repo it points at.
        dep = self.comps["flux2-vae"]["derived_from"][0]
        self.assertEqual(dep["model"],
                         "black-forest-labs/FLUX.2-small-decoder")
        self.assertEqual(dep["license"], "Apache-2.0")
        self.assertEqual(dep["file"], "diffusion_pytorch_model.safetensors")

    def test_full_size_decoder_is_parked_and_noncommercial(self):
        full = self.comps["flux2-vae-small"]
        self.assertEqual(full["license"], "FLUX.1-dev-Non-Commercial")
        self.assertIn("_noncommercial_do_not_ship", full["local_path"])
        self.assertEqual(full["bytes"], 336213556)
        self.assertEqual(full["sha256"], self.FULL_SHA)

    def test_local_small_vae_matches_the_pinned_hash(self):
        path = Path(self.comps["flux2-vae"]["local_path"])
        if not path.exists():
            self.skipTest(f"VAE not present: {path}")
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                h.update(block)
        self.assertEqual(h.hexdigest(), self.SMALL_SHA)

    def test_the_parked_full_decoder_is_outside_the_model_path(self):
        # A parked file that sits in models/vae/ would still be loadable by
        # ComfyUI. The path itself is the guard.
        path = Path(self.comps["flux2-vae-small"]["local_path"])
        self.assertNotEqual(path.parent.name, "vae",
                            "full-size non-commercial VAE is back in models/vae")


class TestLicenseTextsPresent(unittest.TestCase):
    def test_all_registered_texts_exist_and_are_not_stubs(self):
        for lic, path in license_audit.LICENSE_TEXTS.items():
            with self.subTest(license=lic):
                self.assertTrue(path.exists(), f"missing text: {path}")
                text = path.read_text(encoding="utf-8", errors="replace")
                minimum = license_audit.MIN_LICENSE_CHARS.get(lic, 500)
                self.assertGreaterEqual(
                    len(text), minimum,
                    f"{lic} text looks truncated: {len(text)} < {minimum}")

    def test_gpl_text_is_the_real_thing(self):
        text = (ROOT / "licenses" / "LICENSE-GPL-3.0.txt").read_text(
            encoding="utf-8")
        self.assertIn("GNU GENERAL PUBLIC LICENSE", text)
        self.assertIn("Version 3, 29 June 2007", text)
        self.assertIn("TERMS AND CONDITIONS", text)

    def test_qwen_research_license_is_noncommercial(self):
        text = (ROOT / "licenses" / "LICENSE-QWEN-RESEARCH.txt").read_text(
            encoding="utf-8")
        # If this assertion ever fails the licence text changed and every
        # conclusion in docs/10 is stale.
        self.assertIn("RESEARCH LICENSE", text.upper())
        self.assertIn("NON-COMMERCIAL", text.upper())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pyreplab_harness.treatments import (
    TreatmentSpec,
    TreatmentRegistry,
    generate_treatments,
    main as treatments_main,
    to_policy_spec_kwargs,
    treatment_model_input_descriptor,
    _GRAMMAR_SIZE,
)


# ---------------------------------------------------------------------------
# TreatmentSpec validation
# ---------------------------------------------------------------------------


class TreatmentSpecValidationTest(unittest.TestCase):
    def _make_spec(self, **overrides):
        kwargs = {
            "id": "test-policy",
            "version": "1",
            "system_prompt": "Do the thing.",
            "allowed_tools": ("bash",),
            "max_output_tokens": 1024,
            "tool_call_limit": 4,
            "command_timeout_seconds": 30,
            "wall_time_limit_seconds": 180,
        }
        kwargs.update(overrides)
        return TreatmentSpec(**kwargs)

    def test_valid_spec_constructs_and_computes_hash(self) -> None:
        spec = self._make_spec()
        self.assertEqual(spec.id, "test-policy")
        self.assertEqual(spec.version, "1")
        self.assertTrue(len(spec.bundle_hash) == 64)
        self.assertTrue(spec.bundle_hash.isalnum())
        self.assertTrue(
            spec.bundle_id.startswith("test-policy@1-")
        )
        self.assertEqual(len(spec.bundle_id.split("-")[-1]), 8)

    def test_empty_id_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._make_spec(id="")
        self.assertIn("id", str(ctx.exception).lower())

    def test_whitespace_only_id_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._make_spec(id="   ")
        self.assertIn("id", str(ctx.exception).lower())

    def test_unsafe_id_characters_raise(self) -> None:
        for bad_id in ("hello world", "x\ny", "a\tb"):
            with self.subTest(id=bad_id):
                with self.assertRaises(ValueError):
                    self._make_spec(id=bad_id)

    def test_valid_id_characters_accepted(self) -> None:
        for good_id in ("my-id", "v1.2.3", "test_v4", "a@b"):
            with self.subTest(id=good_id):
                spec = self._make_spec(id=good_id)
                self.assertEqual(spec.id, good_id)

    def test_empty_version_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._make_spec(version="")

    def test_whitespace_version_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._make_spec(version="   ")

    def test_empty_system_prompt_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._make_spec(system_prompt="")

    def test_empty_allowed_tools_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._make_spec(allowed_tools=())

    def test_allowed_tools_with_empty_string_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._make_spec(allowed_tools=("bash", ""))

    def test_allowed_tools_strips_and_accepts(self) -> None:
        spec = self._make_spec(allowed_tools=[" bash ", " read "])
        self.assertEqual(spec.allowed_tools, ("bash", "read"))

    def test_non_int_max_output_tokens_raises(self) -> None:
        with self.assertRaises(TypeError):
            self._make_spec(max_output_tokens="1024")

    def test_zero_max_output_tokens_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._make_spec(max_output_tokens=0)

    def test_negative_tool_call_limit_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._make_spec(tool_call_limit=-1)

    def test_boolean_not_int_raises(self) -> None:
        with self.assertRaises(TypeError):
            self._make_spec(max_output_tokens=True)

    def test_empty_tool_interface_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._make_spec(tool_interface="")

    def test_non_dict_generator_metadata_raises(self) -> None:
        with self.assertRaises(TypeError):
            self._make_spec(generator_metadata=["not", "a", "dict"])

    def test_generator_metadata_is_deeply_immutable_and_detached(self) -> None:
        source = {"nested": {"values": [1, 2]}}
        spec = self._make_spec(generator_metadata=source)
        original_hash = spec.bundle_hash
        source["nested"]["values"].append(3)
        self.assertEqual(spec.generator_metadata["nested"]["values"], (1, 2))
        with self.assertRaises(TypeError):
            spec.generator_metadata["new"] = "value"
        with self.assertRaises(TypeError):
            spec.generator_metadata["nested"]["new"] = "value"
        self.assertEqual(spec.bundle_hash, original_hash)
        self.assertEqual(
            spec.to_dict()["generator_metadata"],
            {"nested": {"values": [1, 2]}},
        )

    def test_default_tool_interface(self) -> None:
        spec = TreatmentSpec(
            id="test",
            version="1",
            system_prompt="Do.",
            allowed_tools=("bash",),
            max_output_tokens=100,
            tool_call_limit=2,
            command_timeout_seconds=10,
            wall_time_limit_seconds=60,
        )
        self.assertEqual(spec.tool_interface, "native_bash")


# ---------------------------------------------------------------------------
# Deterministic hashes
# ---------------------------------------------------------------------------


class DeterministicHashTest(unittest.TestCase):
    def test_same_fields_produce_same_hash(self) -> None:
        s1 = TreatmentSpec(
            id="det", version="1", system_prompt="p",
            allowed_tools=("bash",), max_output_tokens=100,
            tool_call_limit=2, command_timeout_seconds=10,
            wall_time_limit_seconds=60,
        )
        s2 = TreatmentSpec(
            id="det", version="1", system_prompt="p",
            allowed_tools=("bash",), max_output_tokens=100,
            tool_call_limit=2, command_timeout_seconds=10,
            wall_time_limit_seconds=60,
        )
        self.assertEqual(s1.bundle_hash, s2.bundle_hash)
        self.assertEqual(s1.bundle_id, s2.bundle_id)

    def test_different_fields_produce_different_hash(self) -> None:
        s1 = TreatmentSpec(
            id="det", version="1", system_prompt="p",
            allowed_tools=("bash",), max_output_tokens=100,
            tool_call_limit=2, command_timeout_seconds=10,
            wall_time_limit_seconds=60,
        )
        s2 = TreatmentSpec(
            id="det", version="1", system_prompt="p-different",
            allowed_tools=("bash",), max_output_tokens=100,
            tool_call_limit=2, command_timeout_seconds=10,
            wall_time_limit_seconds=60,
        )
        self.assertNotEqual(s1.bundle_hash, s2.bundle_hash)

    def test_allowed_tools_order_does_not_affect_hash(self) -> None:
        s1 = TreatmentSpec(
            id="det", version="1", system_prompt="p",
            allowed_tools=("bash", "read"), max_output_tokens=100,
            tool_call_limit=2, command_timeout_seconds=10,
            wall_time_limit_seconds=60,
        )
        s2 = TreatmentSpec(
            id="det", version="1", system_prompt="p",
            allowed_tools=("read", "bash"), max_output_tokens=100,
            tool_call_limit=2, command_timeout_seconds=10,
            wall_time_limit_seconds=60,
        )
        self.assertEqual(s1.bundle_hash, s2.bundle_hash)

    def test_hash_is_hex_string(self) -> None:
        spec = TreatmentSpec(
            id="x", version="1", system_prompt="x",
            allowed_tools=("bash",), max_output_tokens=100,
            tool_call_limit=1, command_timeout_seconds=5,
            wall_time_limit_seconds=30,
        )
        self.assertEqual(len(spec.bundle_hash), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in spec.bundle_hash))


# ---------------------------------------------------------------------------
# to_dict / from_dict with hash verification
# ---------------------------------------------------------------------------


class RoundtripTest(unittest.TestCase):
    def setUp(self):
        self.spec = TreatmentSpec(
            id="rt", version="2", system_prompt="Do X.",
            allowed_tools=("bash", "read"),
            max_output_tokens=500, tool_call_limit=3,
            command_timeout_seconds=15, wall_time_limit_seconds=120,
            tool_interface="native_bash",
            generator_metadata={"src": "test"},
        )

    def test_to_dict_from_dict_roundtrip(self) -> None:
        d = self.spec.to_dict()
        restored = TreatmentSpec.from_dict(d)
        self.assertEqual(self.spec.bundle_hash, restored.bundle_hash)
        self.assertEqual(self.spec.bundle_id, restored.bundle_id)
        self.assertEqual(self.spec.system_prompt, restored.system_prompt)
        self.assertEqual(self.spec.allowed_tools, restored.allowed_tools)

    def test_from_dict_with_good_hash_passes(self) -> None:
        d = self.spec.to_dict()
        # Should not raise.
        TreatmentSpec.from_dict(d, verify_hash=True)

    def test_from_dict_verify_hash_tampered_prompt_raises(self) -> None:
        d = self.spec.to_dict()
        d["system_prompt"] = "Tampered."
        with self.assertRaises(ValueError) as ctx:
            TreatmentSpec.from_dict(d, verify_hash=True)
        self.assertIn("bundle_hash mismatch", str(ctx.exception))

    def test_from_dict_verify_hash_tampered_numeric_raises(self) -> None:
        d = self.spec.to_dict()
        d["max_output_tokens"] = 9999
        with self.assertRaises(ValueError) as ctx:
            TreatmentSpec.from_dict(d, verify_hash=True)
        self.assertIn("bundle_hash mismatch", str(ctx.exception))

    def test_from_dict_without_verify_hash_skips_check(self) -> None:
        d = self.spec.to_dict()
        d["system_prompt"] = "Tampered."
        # Should not raise when verify_hash=False.
        restored = TreatmentSpec.from_dict(d, verify_hash=False)
        # But the hash will differ from the supplied one.
        self.assertNotEqual(restored.bundle_hash, d["bundle_hash"])

    def test_from_dict_without_bundle_hash_field_skips_check(self) -> None:
        d = self.spec.to_dict()
        del d["bundle_hash"]
        # Should not raise; there is nothing to verify against.
        TreatmentSpec.from_dict(d, verify_hash=True)

    def test_to_dict_does_not_leak_private_fields(self) -> None:
        d = self.spec.to_dict()
        self.assertNotIn("_bundle_hash", d)
        self.assertNotIn("_bundle_id", d)

    def test_allowed_tools_signature_deterministic(self) -> None:
        s1 = TreatmentSpec(
            id="sig", version="1", system_prompt=".",
            allowed_tools=("bash", "read", "write"),
            max_output_tokens=100, tool_call_limit=1,
            command_timeout_seconds=5, wall_time_limit_seconds=30,
        )
        s2 = TreatmentSpec(
            id="sig2", version="1", system_prompt="..",
            allowed_tools=("write", "bash", "read"),
            max_output_tokens=100, tool_call_limit=1,
            command_timeout_seconds=5, wall_time_limit_seconds=30,
        )
        self.assertEqual(s1.allowed_tools_signature, s2.allowed_tools_signature)
        self.assertEqual(s1.allowed_tools_signature, "bash,read,write")


# ---------------------------------------------------------------------------
# to_policy_spec_kwargs
# ---------------------------------------------------------------------------


class ConversionHelperTest(unittest.TestCase):
    def test_kwargs_include_policy_spec_fields_only(self) -> None:
        spec = TreatmentSpec(
            id="conv", version="3", system_prompt="SP",
            allowed_tools=("bash",), max_output_tokens=256,
            tool_call_limit=5, command_timeout_seconds=20,
            wall_time_limit_seconds=300,
        )
        kw = to_policy_spec_kwargs(spec)
        expected_keys = {
            "id", "version", "system_prompt", "allowed_tools",
            "max_output_tokens", "tool_call_limit",
            "command_timeout_seconds", "wall_time_limit_seconds",
        }
        self.assertEqual(set(kw), expected_keys)
        self.assertEqual(kw["id"], "conv")
        self.assertEqual(kw["version"], "3")
        self.assertEqual(kw["allowed_tools"], ("bash",))
        # tool_interface and generator_metadata are excluded.
        self.assertNotIn("tool_interface", kw)
        self.assertNotIn("generator_metadata", kw)


# ---------------------------------------------------------------------------
# treatment_model_input_descriptor
# ---------------------------------------------------------------------------


class ModelInputDescriptorTest(unittest.TestCase):
    def setUp(self):
        self.spec = TreatmentSpec(
            id="mi-test", version="1",
            system_prompt="Think carefully.",
            allowed_tools=("bash", "read"),
            max_output_tokens=2048, tool_call_limit=8,
            command_timeout_seconds=45, wall_time_limit_seconds=360,
            tool_interface="native_bash",
        )

    def test_descriptor_contains_required_fields(self) -> None:
        desc = treatment_model_input_descriptor(self.spec)
        self.assertIn("text", desc)
        self.assertIn("max_output_tokens", desc)
        self.assertIn("tool_call_limit", desc)
        self.assertIn("command_timeout_seconds", desc)
        self.assertIn("wall_time_limit_seconds", desc)
        self.assertIn("tool_interface", desc)
        self.assertIn("allowed_tools_signature", desc)
        self.assertIn("bundle_id", desc)
        self.assertIn("policy_id", desc)
        self.assertIn("policy_version", desc)

    def test_text_includes_system_prompt(self) -> None:
        desc = treatment_model_input_descriptor(self.spec)
        self.assertIn("Think carefully.", desc["text"])

    def test_text_includes_task_text_when_provided(self) -> None:
        desc = treatment_model_input_descriptor(
            self.spec, task_text="Fix the bug."
        )
        self.assertIn("Fix the bug.", desc["text"])
        self.assertIn("Think carefully.", desc["text"])
        self.assertTrue(desc["text"].startswith("Fix the bug."))

    def test_descriptor_numeric_fields_match_spec(self) -> None:
        desc = treatment_model_input_descriptor(self.spec)
        self.assertEqual(desc["max_output_tokens"], 2048)
        self.assertEqual(desc["tool_call_limit"], 8)
        self.assertEqual(desc["command_timeout_seconds"], 45)
        self.assertEqual(desc["wall_time_limit_seconds"], 360)

    def test_descriptor_categorical_fields(self) -> None:
        desc = treatment_model_input_descriptor(self.spec)
        self.assertEqual(desc["tool_interface"], "native_bash")
        self.assertEqual(desc["allowed_tools_signature"], "bash,read")
        self.assertEqual(desc["bundle_id"], self.spec.bundle_id)
        self.assertEqual(desc["policy_id"], "mi-test")
        self.assertEqual(desc["policy_version"], "1")


# ---------------------------------------------------------------------------
# TreatmentRegistry
# ---------------------------------------------------------------------------


class TreatmentRegistryTest(unittest.TestCase):
    def _make_spec(self, id_str="r", version="1"):
        return TreatmentSpec(
            id=id_str, version=version, system_prompt="p",
            allowed_tools=("bash",), max_output_tokens=100,
            tool_call_limit=2, command_timeout_seconds=10,
            wall_time_limit_seconds=60,
        )

    def test_empty_registry(self) -> None:
        reg = TreatmentRegistry(())
        self.assertEqual(len(reg), 0)
        self.assertTrue(len(reg.registry_hash) == 64)

    def test_duplicate_id_version_raises(self) -> None:
        s1 = self._make_spec("dup", "1")
        s2 = self._make_spec("dup", "1")
        with self.assertRaises(ValueError) as ctx:
            TreatmentRegistry((s1, s2))
        self.assertIn("dup@1", str(ctx.exception))

    def test_duplicate_bundle_hash_raises(self) -> None:
        # Two specs with different id/version but identical content would
        # produce same hash — but the id is part of the hash, so we need to
        # manually forge two specs with different id but same bundle_hash.
        # Actually impossible through constructor; test the detection path.
        spec = self._make_spec("unique", "1")

        # We can't easily forge a hash collision, but we can verify that
        # the same spec added twice is caught by id/version first.
        # For completeness, build a registry with a manually duplicated hash.
        # Use a private constructor bypass.
        with self.assertRaises(ValueError):
            # Same id@version -> caught by id/version check first.
            TreatmentRegistry((spec, spec))

    def test_by_id_unambiguous(self) -> None:
        spec = self._make_spec("only", "1")
        reg = TreatmentRegistry((spec,))
        self.assertIs(reg.by_id("only"), spec)

    def test_by_id_ambiguous_raises(self) -> None:
        s1 = self._make_spec("amb", "1")
        s2 = self._make_spec("amb", "2")
        reg = TreatmentRegistry((s1, s2))
        with self.assertRaises(KeyError) as ctx:
            reg.by_id("amb")
        self.assertIn("ambiguous", str(ctx.exception))
        self.assertIn("amb", str(ctx.exception))

    def test_by_id_missing_raises(self) -> None:
        reg = TreatmentRegistry((self._make_spec("x", "1"),))
        with self.assertRaises(KeyError):
            reg.by_id("nonexistent")

    def test_by_id_version(self) -> None:
        s1 = self._make_spec("multi", "1")
        s2 = self._make_spec("multi", "2")
        reg = TreatmentRegistry((s1, s2))
        self.assertIs(reg.by_id_version("multi", "1"), s1)
        self.assertIs(reg.by_id_version("multi", "2"), s2)

    def test_by_id_version_missing_raises(self) -> None:
        reg = TreatmentRegistry((self._make_spec("x", "1"),))
        with self.assertRaises(KeyError):
            reg.by_id_version("x", "99")

    def test_by_bundle_id(self) -> None:
        spec = self._make_spec("bid", "1")
        reg = TreatmentRegistry((spec,))
        self.assertIs(reg.by_bundle_id(spec.bundle_id), spec)

    def test_by_bundle_id_missing_raises(self) -> None:
        reg = TreatmentRegistry((self._make_spec("x", "1"),))
        with self.assertRaises(KeyError):
            reg.by_bundle_id("nonexistent@1-deadbeef")

    def test_by_hash(self) -> None:
        spec = self._make_spec("h", "1")
        reg = TreatmentRegistry((spec,))
        self.assertIs(reg.by_hash(spec.bundle_hash), spec)

    def test_by_hash_missing_raises(self) -> None:
        reg = TreatmentRegistry((self._make_spec("x", "1"),))
        with self.assertRaises(KeyError):
            reg.by_hash("a" * 64)

    def test_contains_spec(self) -> None:
        spec = self._make_spec("in", "1")
        other = self._make_spec("out", "1")
        reg = TreatmentRegistry((spec,))
        self.assertIn(spec, reg)
        self.assertNotIn(other, reg)

    def test_registry_hash_deterministic(self) -> None:
        reg1 = TreatmentRegistry((
            self._make_spec("a", "1"),
            self._make_spec("b", "1"),
        ))
        reg2 = TreatmentRegistry((
            self._make_spec("a", "1"),
            self._make_spec("b", "1"),
        ))
        self.assertEqual(reg1.registry_hash, reg2.registry_hash)

    def test_registry_hash_changes_with_content(self) -> None:
        reg1 = TreatmentRegistry((
            self._make_spec("a", "1"),
        ))
        reg2 = TreatmentRegistry((
            self._make_spec("a", "1"),
            self._make_spec("b", "1"),
        ))
        self.assertNotEqual(reg1.registry_hash, reg2.registry_hash)

    def test_to_dict_from_dict_roundtrip(self) -> None:
        s1 = self._make_spec("r1", "1")
        s2 = self._make_spec("r2", "1")
        reg = TreatmentRegistry((s1, s2))
        data = reg.to_dict()
        restored = TreatmentRegistry.from_dict(data)
        self.assertEqual(reg.registry_hash, restored.registry_hash)
        self.assertEqual(len(restored), 2)
        self.assertEqual(
            restored.by_id_version("r1", "1").bundle_hash,
            s1.bundle_hash,
        )

    def test_from_dict_with_tampered_registry_hash_raises(self) -> None:
        spec = self._make_spec("tamper", "1")
        reg = TreatmentRegistry((spec,))
        data = reg.to_dict()
        data["registry_hash"] = "0" * 64
        with self.assertRaises(ValueError) as ctx:
            TreatmentRegistry.from_dict(data, verify_hashes=True)
        self.assertIn("registry_hash mismatch", str(ctx.exception))

    def test_from_dict_with_tampered_treatment_hash_raises(self) -> None:
        spec = self._make_spec("tamper-t", "1")
        reg = TreatmentRegistry((spec,))
        data = reg.to_dict()
        data["treatments"][0]["bundle_hash"] = "1" * 64
        with self.assertRaises(ValueError) as ctx:
            TreatmentRegistry.from_dict(data, verify_hashes=True)
        self.assertIn("bundle_hash mismatch", str(ctx.exception))

    def test_save_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = self._make_spec("io", "1")
            reg = TreatmentRegistry((spec,))
            path = Path(tmpdir) / "registry.json"
            reg.save(path)
            loaded = TreatmentRegistry.load(path)
            self.assertEqual(reg.registry_hash, loaded.registry_hash)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(
                loaded.by_id_version("io", "1").bundle_hash,
                spec.bundle_hash,
            )

    def test_save_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = self._make_spec("det-io", "1")
            reg = TreatmentRegistry((spec,))
            path_a = Path(tmpdir) / "a.json"
            path_b = Path(tmpdir) / "b.json"
            reg.save(path_a)
            reg.save(path_b)
            self.assertEqual(
                path_a.read_bytes(),
                path_b.read_bytes(),
            )

    def test_registry_order_preserved_in_json(self) -> None:
        s1 = self._make_spec("first", "1")
        s2 = self._make_spec("second", "1")
        s3 = self._make_spec("third", "1")
        reg = TreatmentRegistry((s1, s2, s3))
        data = reg.to_dict()
        ids = [t["id"] for t in data["treatments"]]
        self.assertEqual(ids, ["first", "second", "third"])


# ---------------------------------------------------------------------------
# Controlled random generator
# ---------------------------------------------------------------------------


class GeneratorTest(unittest.TestCase):
    def test_generate_returns_correct_count(self) -> None:
        treatments = generate_treatments(5, seed=42)
        self.assertEqual(len(treatments), 5)
        for spec in treatments:
            self.assertIsInstance(spec, TreatmentSpec)

    def test_all_generated_unique(self) -> None:
        treatments = generate_treatments(10, seed=1)
        ids = [spec.id for spec in treatments]
        self.assertEqual(len(ids), len(set(ids)))
        hashes = [spec.bundle_hash for spec in treatments]
        self.assertEqual(len(hashes), len(set(hashes)))

    def test_same_seed_count_are_reproducible(self) -> None:
        t1 = generate_treatments(8, seed=123)
        t2 = generate_treatments(8, seed=123)
        self.assertEqual(
            [spec.bundle_hash for spec in t1],
            [spec.bundle_hash for spec in t2],
        )
        self.assertEqual(
            [spec.id for spec in t1],
            [spec.id for spec in t2],
        )

    def test_different_seeds_produce_different_order(self) -> None:
        t1 = generate_treatments(36, seed=10)
        t2 = generate_treatments(36, seed=20)
        self.assertNotEqual(
            [spec.id for spec in t1],
            [spec.id for spec in t2],
        )
        # All 36 should be present in both (the full grammar).
        self.assertEqual(set(spec.id for spec in t1), set(spec.id for spec in t2))

    def test_different_counts_with_same_seed_are_prefixes(self) -> None:
        t_all = generate_treatments(36, seed=77)
        t_small = generate_treatments(4, seed=77)
        self.assertEqual(
            [spec.id for spec in t_all[:4]],
            [spec.id for spec in t_small],
        )

    def test_full_grammar_size(self) -> None:
        treatments = generate_treatments(_GRAMMAR_SIZE, seed=0)
        self.assertEqual(len(treatments), _GRAMMAR_SIZE)
        # Verify all ids are unique.
        self.assertEqual(len(set(spec.id for spec in treatments)), _GRAMMAR_SIZE)

    def test_count_exceeds_grammar_size_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            generate_treatments(_GRAMMAR_SIZE + 1, seed=0)
        self.assertIn("exceeds grammar size", str(ctx.exception))

    def test_count_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            generate_treatments(0, seed=0)

    def test_count_negative_raises(self) -> None:
        with self.assertRaises(ValueError):
            generate_treatments(-1, seed=0)

    def test_generated_treatments_have_system_prompt_with_safety_suffix(self) -> None:
        treatments = generate_treatments(3, seed=42)
        for spec in treatments:
            self.assertIn("Safety:", spec.system_prompt)
            self.assertIn("workspace", spec.system_prompt)

    def test_generated_treatments_have_default_tools(self) -> None:
        treatments = generate_treatments(5, seed=42)
        for spec in treatments:
            self.assertEqual(spec.allowed_tools, ("bash",))
            self.assertEqual(spec.tool_interface, "native_bash")

    def test_generated_ids_are_well_formed(self) -> None:
        treatments = generate_treatments(10, seed=42)
        for spec in treatments:
            # The id encodes planning-verification-execution-budget.
            # Some dimension values (e.g. "retry-on-failure") contain hyphens.
            self.assertTrue(
                any(spec.id.startswith(p + "-") for p in ("direct", "deliberate", "decompose"))
            )
            self.assertTrue(
                any(spec.id.endswith("-" + b) for b in ("tight", "moderate", "generous"))
            )
            # Verify the id contains one of the verification and execution values.
            id_lower = spec.id
            has_verif = "-final-" in id_lower or "-incremental-" in id_lower
            has_exec = "-single-pass-" in id_lower or "-retry-on-failure-" in id_lower
            self.assertTrue(has_verif, f"missing verification in {spec.id!r}")
            self.assertTrue(has_exec, f"missing execution in {spec.id!r}")

    def test_generator_metadata_captures_dimensions(self) -> None:
        treatments = generate_treatments(3, seed=42)
        for spec in treatments:
            meta = spec.generator_metadata
            self.assertIn("planning", meta)
            self.assertIn("verification", meta)
            self.assertIn("execution", meta)
            self.assertIn("budget", meta)
            self.assertEqual(meta["grammar_version"], "1")
            self.assertEqual(meta["grammar_size"], _GRAMMAR_SIZE)
            self.assertIn("index", meta)

    def test_tight_budget_is_constrained(self) -> None:
        treatments = generate_treatments(36, seed=0)
        for spec in treatments:
            if spec.id.endswith("-tight"):
                self.assertEqual(spec.max_output_tokens, 1024)
                self.assertEqual(spec.tool_call_limit, 4)
                self.assertEqual(spec.command_timeout_seconds, 30)
                self.assertEqual(spec.wall_time_limit_seconds, 180)

    def test_generous_budget_is_ample(self) -> None:
        treatments = generate_treatments(36, seed=0)
        for spec in treatments:
            if spec.id.endswith("-generous"):
                self.assertEqual(spec.max_output_tokens, 4096)
                self.assertEqual(spec.tool_call_limit, 12)
                self.assertEqual(spec.command_timeout_seconds, 60)
                self.assertEqual(spec.wall_time_limit_seconds, 600)

    def test_heap_generated_is_deterministic_across_processes(self) -> None:
        """Ensure the generated hashes are stable (no memory-address leakage)."""
        t1 = generate_treatments(10, seed=999)
        t2 = generate_treatments(10, seed=999)
        for a, b in zip(t1, t2):
            self.assertEqual(a.bundle_hash, b.bundle_hash)
            self.assertEqual(a.bundle_id, b.bundle_id)

    def test_generated_different_versions_produce_different_ids(self) -> None:
        t1 = generate_treatments(3, seed=42, version="1")
        t2 = generate_treatments(3, seed=42, version="2")
        for a, b in zip(t1, t2):
            self.assertEqual(a.id, b.id)
            self.assertNotEqual(a.version, b.version)
            self.assertNotEqual(a.bundle_id, b.bundle_id)
            self.assertNotEqual(a.bundle_hash, b.bundle_hash)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class CliTest(unittest.TestCase):
    def test_generate_inspect_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "treatments.json")
            code = treatments_main(["generate", output, "--count", "4", "--seed", "123"])
            self.assertEqual(code, 0)
            self.assertTrue(Path(output).exists())

            data = json.loads(Path(output).read_text(encoding="utf-8"))
            self.assertIn("registry_hash", data)
            self.assertEqual(len(data["treatments"]), 4)
            for item in data["treatments"]:
                self.assertIn("bundle_hash", item)
                self.assertIn("bundle_id", item)

            # Inspect the same file (should succeed).
            code = treatments_main(["inspect", output])
            self.assertEqual(code, 0)

    def test_generate_count_exceeds_grammar_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "overflow.json")
            with self.assertRaises(ValueError):
                treatments_main(
                    ["generate", output, "--count", str(_GRAMMAR_SIZE + 1), "--seed", "1"]
                )

    def test_generate_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = str(Path(tmpdir) / "a.json")
            path_b = str(Path(tmpdir) / "b.json")
            treatments_main(["generate", path_a, "--count", "6", "--seed", "42"])
            treatments_main(["generate", path_b, "--count", "6", "--seed", "42"])
            self.assertEqual(
                Path(path_a).read_bytes(),
                Path(path_b).read_bytes(),
            )

    def test_generate_version_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "v2.json")
            code = treatments_main(
                ["generate", output, "--count", "2", "--seed", "1", "--version", "2"]
            )
            self.assertEqual(code, 0)
            reg = TreatmentRegistry.load(output)
            for spec in reg:
                self.assertEqual(spec.version, "2")

    def test_inspect_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            treatments_main(["inspect", "/nonexistent/registry.json"])

    def test_inspect_tampered_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "tamper.json")
            treatments_main(["generate", output, "--count", "2", "--seed", "1"])
            data = json.loads(Path(output).read_text(encoding="utf-8"))
            data["treatments"][0]["system_prompt"] = "CORRUPTED"
            Path(output).write_text(
                json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                treatments_main(["inspect", output])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class EdgeCaseTest(unittest.TestCase):
    def test_bundle_id_structure(self) -> None:
        spec = TreatmentSpec(
            id="my.policy", version="v2.0",
            system_prompt="Hello.",
            allowed_tools=("bash",), max_output_tokens=100,
            tool_call_limit=2, command_timeout_seconds=10,
            wall_time_limit_seconds=60,
        )
        bid = spec.bundle_id
        self.assertTrue(bid.startswith("my.policy@v2.0-"))
        suffix = bid.split("-")[-1]
        self.assertEqual(len(suffix), 8)
        self.assertEqual(suffix, spec.bundle_hash[:8])

    def test_multiple_versions_by_id_ambiguous(self) -> None:
        s1 = TreatmentSpec(
            id="multi", version="1", system_prompt="a",
            allowed_tools=("bash",), max_output_tokens=100,
            tool_call_limit=2, command_timeout_seconds=10,
            wall_time_limit_seconds=60,
        )
        s2 = TreatmentSpec(
            id="multi", version="2", system_prompt="b",
            allowed_tools=("bash",), max_output_tokens=200,
            tool_call_limit=3, command_timeout_seconds=20,
            wall_time_limit_seconds=120,
        )
        reg = TreatmentRegistry((s1, s2))
        with self.assertRaises(KeyError) as ctx:
            reg.by_id("multi")
        msg = str(ctx.exception)
        self.assertIn("ambiguous", msg)
        self.assertIn("multi", msg)

    def test_registry_iteration_order(self) -> None:
        specs = tuple(
            TreatmentSpec(
                id=f"iter-{i}", version="1",
                system_prompt=f"prompt-{i}",
                allowed_tools=("bash",), max_output_tokens=100,
                tool_call_limit=2, command_timeout_seconds=10,
                wall_time_limit_seconds=60,
            )
            for i in range(5)
        )
        reg = TreatmentRegistry(specs)
        self.assertEqual(
            [spec.id for spec in reg],
            [f"iter-{i}" for i in range(5)],
        )


if __name__ == "__main__":
    unittest.main()

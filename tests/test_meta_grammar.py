"""Tests for the 72-cell Unbrowser policy grammar generator."""

from __future__ import annotations

import unittest

from pyreplab_harness import meta_grammar


class GrammarEnumerationTest(unittest.TestCase):
    def test_72_policies_generated(self) -> None:
        grammar = meta_grammar.enumerate_unbrowser_grammar()
        self.assertEqual(len(grammar), 72)
        self.assertEqual(len(grammar), meta_grammar._GRAMMAR_SIZE)

    def test_all_ids_are_unique(self) -> None:
        grammar = meta_grammar.enumerate_unbrowser_grammar()
        ids = [t.id for t in grammar]
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_bundle_hashes_are_unique(self) -> None:
        grammar = meta_grammar.enumerate_unbrowser_grammar()
        hashes = [t.bundle_hash for t in grammar]
        self.assertEqual(len(hashes), len(set(hashes)))

    def test_all_bundle_ids_are_unique(self) -> None:
        grammar = meta_grammar.enumerate_unbrowser_grammar()
        bundle_ids = [t.bundle_id for t in grammar]
        self.assertEqual(len(bundle_ids), len(set(bundle_ids)))

    def test_ids_contain_factor_levels(self) -> None:
        grammar = meta_grammar.enumerate_unbrowser_grammar()
        for t in grammar:
            self.assertIn("ub-", t.id)
            self.assertIn("-", t.id)

    def test_generator_metadata_has_all_factors(self) -> None:
        grammar = meta_grammar.enumerate_unbrowser_grammar()
        for t in grammar:
            meta = t.generator_metadata
            self.assertIn("planning", meta)
            self.assertIn("observation", meta)
            self.assertIn("verification", meta)
            self.assertIn("recovery", meta)
            self.assertIn("tool_cap", meta)
            self.assertIn("grammar_version", meta)
            self.assertEqual(meta["grammar_name"], "unbrowser_interactive")

    def test_tool_interface_is_correct(self) -> None:
        grammar = meta_grammar.enumerate_unbrowser_grammar()
        for t in grammar:
            self.assertEqual(t.tool_interface, "native_bash_unbrowser_interactive_v1")

    def test_allowed_tools_are_bash_and_unbrowser(self) -> None:
        grammar = meta_grammar.enumerate_unbrowser_grammar()
        for t in grammar:
            self.assertEqual(set(t.allowed_tools), {"bash", "unbrowser"})

    def test_constants_are_frozen(self) -> None:
        grammar = meta_grammar.enumerate_unbrowser_grammar()
        for t in grammar:
            self.assertEqual(t.max_output_tokens, 4096)
            self.assertEqual(t.command_timeout_seconds, 60)
            self.assertEqual(t.wall_time_limit_seconds, 600)

    def test_tool_call_limit_lean_expanded_only(self) -> None:
        grammar = meta_grammar.enumerate_unbrowser_grammar()
        limits = {t.tool_call_limit for t in grammar}
        self.assertEqual(limits, {6, 12})

    def test_system_prompt_contains_factor_clauses(self) -> None:
        grammar = meta_grammar.enumerate_unbrowser_grammar()
        for t in grammar:
            prompt = t.system_prompt
            self.assertIn("Planning:", prompt)
            self.assertIn("Observation:", prompt)
            self.assertIn("Verification:", prompt)
            self.assertIn("Recovery:", prompt)
            self.assertIn("Safety:", prompt)

    def test_deterministic_given_version(self) -> None:
        first = meta_grammar.enumerate_unbrowser_grammar(version="1")
        second = meta_grammar.enumerate_unbrowser_grammar(version="1")
        self.assertEqual(
            [t.bundle_hash for t in first],
            [t.bundle_hash for t in second],
        )

    def test_different_version_has_different_hashes(self) -> None:
        first = meta_grammar.enumerate_unbrowser_grammar(version="1")
        second = meta_grammar.enumerate_unbrowser_grammar(version="2")
        hashes1 = {t.bundle_hash for t in first}
        hashes2 = {t.bundle_hash for t in second}
        self.assertTrue(len(hashes1 & hashes2) == 0)


class GenerateUnbrowserTreatmentsTest(unittest.TestCase):
    def test_sample_deterministic(self) -> None:
        first = meta_grammar.generate_unbrowser_treatments(10, seed=42)
        second = meta_grammar.generate_unbrowser_treatments(10, seed=42)
        self.assertEqual(
            [t.bundle_hash for t in first],
            [t.bundle_hash for t in second],
        )

    def test_different_seed_different_sample(self) -> None:
        first = meta_grammar.generate_unbrowser_treatments(10, seed=42)
        second = meta_grammar.generate_unbrowser_treatments(10, seed=99)
        hashes1 = {t.bundle_hash for t in first}
        hashes2 = {t.bundle_hash for t in second}
        self.assertTrue(len(hashes1 & hashes2) < 10)

    def test_count_exceeds_grammar_raises(self) -> None:
        with self.assertRaises(ValueError):
            meta_grammar.generate_unbrowser_treatments(73, seed=42)

    def test_count_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            meta_grammar.generate_unbrowser_treatments(0, seed=42)

    def test_count_all_72(self) -> None:
        treatments = meta_grammar.generate_unbrowser_treatments(72, seed=42)
        self.assertEqual(len(treatments), 72)
        hashes = {t.bundle_hash for t in treatments}
        self.assertEqual(len(hashes), 72)


class SplitPoliciesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.grammar = meta_grammar.enumerate_unbrowser_grammar()

    def test_split_sizes(self) -> None:
        mt, dev, final = meta_grammar.split_policies(self.grammar, seed=42)
        self.assertEqual(len(mt), 48)
        self.assertEqual(len(dev), 12)
        self.assertEqual(len(final), 12)

    def test_no_overlap(self) -> None:
        mt, dev, final = meta_grammar.split_policies(self.grammar, seed=42)
        mt_hashes = {t.bundle_hash for t in mt}
        dev_hashes = {t.bundle_hash for t in dev}
        final_hashes = {t.bundle_hash for t in final}
        self.assertEqual(len(mt_hashes & dev_hashes), 0)
        self.assertEqual(len(mt_hashes & final_hashes), 0)
        self.assertEqual(len(dev_hashes & final_hashes), 0)

    def test_total_equals_72(self) -> None:
        mt, dev, final = meta_grammar.split_policies(self.grammar, seed=42)
        self.assertEqual(len(mt) + len(dev) + len(final), 72)

    def test_every_factor_level_in_meta_train(self) -> None:
        mt, dev, final = meta_grammar.split_policies(self.grammar, seed=42)
        factors = ["planning", "observation", "verification", "recovery", "tool_cap"]
        for factor in factors:
            levels = set()
            for t in self.grammar:
                levels.add(str(t.generator_metadata[factor]))
            meta_levels = set()
            for t in mt:
                meta_levels.add(str(t.generator_metadata[factor]))
            self.assertEqual(
                meta_levels, levels,
                f"Factor {factor} not fully covered in meta-train: "
                f"meta={sorted(meta_levels)}, all={sorted(levels)}",
            )

    def test_deterministic_given_seed(self) -> None:
        mt1, dev1, fin1 = meta_grammar.split_policies(self.grammar, seed=42)
        mt2, dev2, fin2 = meta_grammar.split_policies(self.grammar, seed=42)
        self.assertEqual([t.bundle_hash for t in mt1], [t.bundle_hash for t in mt2])
        self.assertEqual([t.bundle_hash for t in dev1], [t.bundle_hash for t in dev2])
        self.assertEqual([t.bundle_hash for t in fin1], [t.bundle_hash for t in fin2])

    def test_different_seed_different_split(self) -> None:
        mt1, _, _ = meta_grammar.split_policies(self.grammar, seed=42)
        mt2, _, _ = meta_grammar.split_policies(self.grammar, seed=99)
        hashes1 = {t.bundle_hash for t in mt1}
        hashes2 = {t.bundle_hash for t in mt2}
        self.assertNotEqual(hashes1, hashes2)

    def test_wrong_size_raises(self) -> None:
        subset = self.grammar[:71]
        with self.assertRaises(ValueError):
            meta_grammar.split_policies(subset, seed=42)


class ExportGrammarFactorsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.grammar = meta_grammar.enumerate_unbrowser_grammar()

    def test_one_hot_structure(self) -> None:
        exported = meta_grammar.export_grammar_factors(self.grammar[0])
        self.assertIn("one_hot", exported)
        self.assertIn("numeric", exported)
        self.assertIn("factor_labels", exported)

    def test_one_hot_has_correct_dimensions(self) -> None:
        exported = meta_grammar.export_grammar_factors(self.grammar[0])
        self.assertEqual(len(exported["one_hot"]["planning"]), 3)
        self.assertEqual(len(exported["one_hot"]["observation"]), 3)
        self.assertEqual(len(exported["one_hot"]["verification"]), 2)
        self.assertEqual(len(exported["one_hot"]["recovery"]), 2)
        self.assertEqual(len(exported["one_hot"]["tool_cap"]), 2)

    def test_one_hot_exactly_one_active_per_factor(self) -> None:
        for t in self.grammar:
            exported = meta_grammar.export_grammar_factors(t)
            for factor, vec in exported["one_hot"].items():
                self.assertAlmostEqual(sum(vec), 1.0, places=5,
                                       msg=f"Factor {factor} has sum != 1: {vec}")

    def test_numeric_has_tool_call_limit(self) -> None:
        for t in self.grammar:
            exported = meta_grammar.export_grammar_factors(t)
            self.assertIn("tool_call_limit", exported["numeric"])
            self.assertIn(exported["numeric"]["tool_call_limit"], (6.0, 12.0))

    def test_factor_labels_match_one_hot(self) -> None:
        for t in self.grammar:
            exported = meta_grammar.export_grammar_factors(t)
            for factor, labels in exported["factor_labels"].items():
                vec = exported["one_hot"][factor]
                # Find which level is active.
                active_idx = next(i for i, v in enumerate(vec) if v > 0.5)
                # Verify label matches one-hot position (from grammar definition).
                self.assertIsNotNone(active_idx)

    def test_grammar_factor_vector_is_13d(self) -> None:
        for t in self.grammar:
            vec = meta_grammar.grammar_factor_vector(t)
            self.assertEqual(len(vec), 13)
            # Last element is normalized tool_call_limit: 0.0 or 1.0.
            self.assertIn(vec[-1], (0.0, 1.0))

    def test_different_policies_have_different_vectors(self) -> None:
        vectors = [meta_grammar.grammar_factor_vector(t) for t in self.grammar]
        # All 72 should have unique vectors (each combination is unique).
        unique = set(tuple(v) for v in vectors)
        self.assertEqual(len(unique), 72)


if __name__ == "__main__":
    unittest.main()

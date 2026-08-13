from __future__ import annotations

import unittest

from pyreplab_harness.structural_probe import (
    FEATURE_CAPS,
    FEATURE_KEYS,
    MAX_SOURCE_BYTES,
    MECHANISM,
    SCHEMA_VERSION,
    StructuralProbeError,
    audit_features,
    audit_receipt,
    audit_result,
    canonical_feature_sha256,
    cap_features,
    parse_features,
    structural_probe,
)

# A representative document exercising every feature family. Hand-counted below.
REPRESENTATIVE = """
<html>
  <head><title>Ignored title</title></head>
  <body>
    <h1>Hello</h1>
    <a href="/x" class="link" id="top">A link</a>
    <table id="t1">
      <tr><th>H1</th><th>H2</th><th>H3</th></tr>
      <tr><td>a</td><td>b</td><td>c</td></tr>
    </table>
    <form method="post" action="/submit">
      <label>Name <input type="text" name="n" required></label>
      <input type="password" name="p">
      <select name="s"><option>1</option><option>2</option></select>
      <textarea name="t"></textarea>
      <button type="submit">Go</button>
    </form>
    <form>
      <input type="checkbox" name="c">
    </form>
  </body>
</html>
"""


class StructuralProbeTest(unittest.TestCase):
    def test_representative_document_counts(self) -> None:
        features = structural_probe(REPRESENTATIVE)["features"]
        self.assertEqual(features["element_count"], 26)
        self.assertEqual(features["max_dom_depth"], 5)
        self.assertEqual(features["table_count"], 1)
        self.assertEqual(features["table_row_count"], 2)
        self.assertEqual(features["table_cell_count"], 6)
        self.assertEqual(features["max_table_columns"], 3)
        self.assertEqual(features["form_count"], 2)
        self.assertEqual(features["control_count"], 6)
        self.assertEqual(features["required_control_count"], 1)
        self.assertEqual(features["get_form_count"], 1)
        self.assertEqual(features["post_form_count"], 1)
        self.assertEqual(features["text_input_count"], 2)
        self.assertEqual(features["select_count"], 1)
        self.assertEqual(features["textarea_count"], 1)
        self.assertEqual(features["button_count"], 1)
        self.assertEqual(features["anchor_count"], 1)

    def test_allowlist_exactly_matches_schema(self) -> None:
        self.assertEqual(
            list(structural_probe(REPRESENTATIVE)["features"].keys()),
            list(FEATURE_KEYS),
        )
        self.assertEqual(
            set(FEATURE_CAPS),
            set(FEATURE_KEYS),
        )

    def test_form_method_classification(self) -> None:
        html = (
            '<form method="POST"></form>'
            '<form method="get"></form>'
            "<form></form>"
            '<form method="dialog"></form>'
            "<form METHOD=post></form>"
        )
        features = structural_probe(html)["features"]
        self.assertEqual(features["form_count"], 5)
        self.assertEqual(features["get_form_count"], 2)  # get + default
        self.assertEqual(features["post_form_count"], 2)  # POST + post

    def test_input_type_and_button_classification(self) -> None:
        html = (
            '<input type="text">'
            '<input type="PASSWORD">'
            '<input type="checkbox">'
            '<input type="radio">'
            '<input type="submit">'
            '<input type="reset">'
            '<input type="button">'
            '<input type="image">'
            "<input>"  # default text
            "<button></button>"
            '<button type="submit"></button>'
        )
        features = structural_probe(html)["features"]
        self.assertEqual(features["control_count"], 11)
        self.assertEqual(features["text_input_count"], 3)  # text, password, default
        self.assertEqual(features["button_count"], 6)  # 4 button-inputs + 2 <button>
        self.assertEqual(features["required_control_count"], 0)

    def test_required_control_count(self) -> None:
        html = (
            '<input type="text" required>'
            '<input type="text" required="required">'
            "<select required></select>"
            "<textarea required></textarea>"
            "<input type='text'>"
        )
        features = structural_probe(html)["features"]
        self.assertEqual(features["required_control_count"], 4)
        self.assertEqual(features["control_count"], 5)

    def test_max_table_columns_takes_maximum_row(self) -> None:
        html = (
            "<table><tr><td>1</td><td>2</td></tr>"
            "<tr><td>1</td><td>2</td><td>3</td><td>4</td></tr></table>"
        )
        features = structural_probe(html)["features"]
        self.assertEqual(features["table_row_count"], 2)
        self.assertEqual(features["table_cell_count"], 6)
        self.assertEqual(features["max_table_columns"], 4)

    def test_nested_table_columns(self) -> None:
        html = (
            "<table><tr><td>outer</td><td>"
            "<table><tr><td>a</td><td>b</td><td>c</td></tr></table>"
            "</td></tr></table>"
        )
        features = structural_probe(html)["features"]
        self.assertEqual(features["table_count"], 2)
        self.assertEqual(features["table_row_count"], 2)
        self.assertEqual(features["table_cell_count"], 5)
        self.assertEqual(features["max_table_columns"], 3)

    def test_void_elements_do_not_inflate_depth(self) -> None:
        features = structural_probe(
            "<div><span><br><img src='x'><input type='text'></span></div>"
        )["features"]
        self.assertEqual(features["element_count"], 5)
        self.assertEqual(features["max_dom_depth"], 2)

    def test_empty_input_is_all_zeros(self) -> None:
        features = structural_probe("")["features"]
        self.assertEqual(set(features.values()), {0})
        self.assertEqual(list(features.keys()), list(FEATURE_KEYS))


class StructuralProbePrivacyTest(unittest.TestCase):
    def test_same_structure_different_text_identical_features(self) -> None:
        one = "<table><tr><td>apple</td><td>banana</td></tr></table>"
        two = "<table><tr><td>zebra</td><td>quokka</td></tr></table>"
        r1 = structural_probe(one)
        r2 = structural_probe(two)
        self.assertEqual(r1["features"], r2["features"])
        self.assertEqual(
            r1["receipt"]["canonical_feature_sha256"],
            r2["receipt"]["canonical_feature_sha256"],
        )
        self.assertNotEqual(r1["receipt"]["source_sha256"], r2["receipt"]["source_sha256"])

    def test_attributes_urls_labels_do_not_change_features(self) -> None:
        base = (
            "<a href='/a' class='x' id='y'>text</a>"
            "<input type='text' name='n' value='v' placeholder='p' aria-label='l'>"
        )
        leaky = (
            "<a href='https://evil.example/secret?token=123' class='zz' id='w'>DIFFERENT</a>"
            "<input type='text' name='q' value='vvv' placeholder='pp' aria-label='ll'>"
        )
        self.assertEqual(
            structural_probe(base)["features"],
            structural_probe(leaky)["features"],
        )

    def test_features_are_ints_only_and_audit_is_clean(self) -> None:
        result = structural_probe(REPRESENTATIVE)
        for key, value in result["features"].items():
            self.assertIsInstance(value, int)
            self.assertNotIsInstance(value, bool)
        self.assertEqual(audit_features(result["features"]), [])
        self.assertEqual(audit_receipt(result["receipt"], result["features"]), [])
        self.assertEqual(audit_result(result), [])

    def test_audit_flags_forbidden_and_extra_keys(self) -> None:
        clean = structural_probe(REPRESENTATIVE)["features"]
        bad = dict(clean)
        bad["page_url"] = "https://example.com/secret"
        bad["label"] = "confidential"
        violations = audit_features(bad)
        self.assertTrue(any("page_url" in v for v in violations))
        self.assertTrue(any("label" in v for v in violations))

    def test_audit_flags_missing_keys(self) -> None:
        clean = structural_probe(REPRESENTATIVE)["features"]
        bad = dict(clean)
        del bad["element_count"]
        violations = audit_features(bad)
        self.assertTrue(any("element_count" in v for v in violations))

    def test_audit_flags_out_of_bound_and_non_integer_values(self) -> None:
        clean = structural_probe(REPRESENTATIVE)["features"]
        for key, value in (("anchor_count", -1), ("anchor_count", 10 ** 12)):
            bad = dict(clean)
            bad["anchor_count"] = value
            self.assertTrue(audit_features(bad), "expected violation for %r" % value)
        bad = dict(clean)
        bad["anchor_count"] = True
        self.assertTrue(any("integer" in v for v in audit_features(bad)))
        bad = dict(clean)
        bad["anchor_count"] = "3"
        self.assertTrue(any("integer" in v for v in audit_features(bad)))

    def test_receipt_audit_rejects_tampering(self) -> None:
        result = structural_probe(REPRESENTATIVE)
        features = result["features"]
        base = result["receipt"]

        tampered = dict(base)
        tampered["canonical_feature_sha256"] = "0" * 64
        self.assertTrue(audit_receipt(tampered, features))

        tampered = dict(base)
        tampered["delivered"] = False
        self.assertTrue(audit_receipt(tampered, features))

        tampered = dict(base)
        tampered["schema_version"] = "evil-schema"
        self.assertTrue(audit_receipt(tampered, features))

        tampered = dict(base)
        tampered["mechanism"] = "evil-mechanism"
        self.assertTrue(audit_receipt(tampered, features))

        tampered = dict(base)
        tampered["source_sha256"] = "zz" * 32
        self.assertTrue(audit_receipt(tampered, features))

    def test_receipt_schema_constants(self) -> None:
        result = structural_probe(REPRESENTATIVE)
        receipt = result["receipt"]
        self.assertEqual(receipt["schema_version"], SCHEMA_VERSION)
        self.assertEqual(receipt["mechanism"], MECHANISM)
        self.assertEqual(
            receipt["schema_version"], "pyreplab-public-html-structural-probe-v1"
        )
        self.assertEqual(
            receipt["mechanism"], "controller_owned_public_html_structural_probe"
        )
        self.assertIs(receipt["delivered"], True)
        self.assertEqual(receipt["source_bytes"], len(REPRESENTATIVE.encode("utf-8")))


class StructuralProbeBoundaryTest(unittest.TestCase):
    def test_bytes_and_text_equivalence(self) -> None:
        html = "<div><a href='/x'>link</a><form method='post'><input type='text'></form></div>"
        r_text = structural_probe(html)
        r_bytes = structural_probe(html.encode("utf-8"))
        self.assertEqual(r_text, r_bytes)
        self.assertEqual(r_text["receipt"]["source_bytes"], len(html.encode("utf-8")))

    def test_determinism_across_replays(self) -> None:
        first = structural_probe(REPRESENTATIVE)
        for _ in range(5):
            self.assertEqual(first, structural_probe(REPRESENTATIVE))

    def test_canonicalization_is_order_independent(self) -> None:
        features = structural_probe(REPRESENTATIVE)["features"]
        shuffled = dict(reversed(list(features.items())))
        self.assertEqual(
            canonical_feature_sha256(features),
            canonical_feature_sha256(shuffled),
        )

    def test_malformed_html_does_not_raise_and_stays_deterministic(self) -> None:
        fragments = [
            "<div><span></div>",
            "<table><tr><td>x",
            "</div></body></html>",
            "<p>unclosed",
            "<a href='unterminated>",
            "<div><<<<",
            "<form method='post'",
            "<script>if (a < b) { x = \"</div>\"; }</script><div>ok</div>",
            "<textarea><b>raw text</b></textarea>",
            "<!DOCTYPE html><!-- comment --><div>?",
        ]
        for fragment in fragments:
            result = structural_probe(fragment)
            self.assertEqual(audit_result(result), [], "fragment: %r" % fragment)
            self.assertEqual(result, structural_probe(fragment))

    def test_oversized_input_is_rejected(self) -> None:
        html = "<div>" * 10_000
        with self.assertRaises(StructuralProbeError) as ctx:
            structural_probe(html, max_source_bytes=100)
        self.assertEqual(ctx.exception.code, "input_oversized")
        self.assertGreater(MAX_SOURCE_BYTES, 0)

    def test_invalid_utf8_bytes_are_rejected(self) -> None:
        with self.assertRaises(StructuralProbeError) as ctx:
            structural_probe(b"\xff\xfe\xc3\x28")
        self.assertEqual(ctx.exception.code, "undecodable_utf8")

    def test_invalid_input_type_is_rejected(self) -> None:
        with self.assertRaises(StructuralProbeError) as ctx:
            structural_probe(123)  # type: ignore[arg-type]
        self.assertEqual(ctx.exception.code, "invalid_input_type")


class StructuralProbeCapTest(unittest.TestCase):
    def test_cap_features_clamps_to_frozen_bounds(self) -> None:
        raw = {key: FEATURE_CAPS[key] + 5 for key in FEATURE_KEYS}
        capped = cap_features(raw)
        self.assertEqual(
            capped,
            {key: FEATURE_CAPS[key] for key in FEATURE_KEYS},
        )

    def test_cap_features_drops_unknown_and_preserves_order(self) -> None:
        raw = {"anchor_count": 3, "element_count": 1, "leaky_url": 99}
        capped = cap_features(raw)
        self.assertEqual(list(capped.keys()), list(FEATURE_KEYS))
        self.assertEqual(capped["anchor_count"], 3)
        self.assertEqual(capped["element_count"], 1)
        self.assertNotIn("leaky_url", capped)

    def test_structural_probe_applies_caps_end_to_end(self) -> None:
        # table_count cap is small enough (10_000) to exercise directly.
        html = "<table></table>" * (FEATURE_CAPS["table_count"] + 1)
        features = structural_probe(html)["features"]
        self.assertEqual(features["table_count"], FEATURE_CAPS["table_count"])

    def test_override_caps_for_testing(self) -> None:
        html = "<a>1</a><a>2</a><a>3</a>"
        features = structural_probe(html, feature_caps={"anchor_count": 2})["features"]
        self.assertEqual(features["anchor_count"], 2)
        # Non-overridden keys still use the frozen caps.
        self.assertEqual(features["element_count"], 3)


class ParseFeaturesHelperTest(unittest.TestCase):
    def test_parse_features_returns_features_only(self) -> None:
        features = parse_features(REPRESENTATIVE)
        self.assertEqual(list(features.keys()), list(FEATURE_KEYS))
        self.assertEqual(features["table_count"], 1)
        self.assertEqual(audit_features(features), [])


if __name__ == "__main__":
    unittest.main()

"""
Unit Tests for Model Export Module
Validates GBDT clean JSON export and schema compatibility with GbdtModelJson.
"""

import os
import sys
import json
import tempfile
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.export.export_json import export_gbdt_to_clean_json


class TestExportModule(unittest.TestCase):

    def setUp(self):
        self.sample_model_path = os.path.join(
            PROJECT_ROOT, "saved_models", "multilingual", "gbdt_universal.pkl"
        )
        if not os.path.exists(self.sample_model_path):
            self.sample_model_path = os.path.join(
                PROJECT_ROOT, "saved_models", "gbdt_model.pkl"
            )

    def test_export_gbdt_clean_json_schema(self):
        if not os.path.exists(self.sample_model_path):
            self.skipTest(f"Model file not found at {self.sample_model_path}")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            dump = export_gbdt_to_clean_json(
                model_input=self.sample_model_path,
                output_path=tmp_path,
                name="test_gbdt",
                version="2.0.0",
                minify=True
            )

            # Check return dictionary schema
            self.assertEqual(dump["name"], "test_gbdt")
            self.assertEqual(dump["version"], "2.0.0")
            self.assertIn("num_features", dump)
            self.assertIn("feature_names", dump)
            self.assertIn("tree_info", dump)
            self.assertGreater(len(dump["tree_info"]), 0)

            # Verify saved file exists and is valid JSON
            self.assertTrue(os.path.exists(tmp_path))
            with open(tmp_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["name"], "test_gbdt")

            # Check node structure
            tree0 = loaded["tree_info"][0]["tree_structure"]
            self.assertTrue("leaf_value" in tree0 or "split_feature" in tree0)
            if "split_feature" in tree0:
                self.assertIn("threshold", tree0)
                self.assertIn("left_child", tree0)
                self.assertIn("right_child", tree0)
                # Ensure extraneous fields like split_gain are removed
                self.assertNotIn("split_gain", tree0)
                self.assertNotIn("internal_value", tree0)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == '__main__':
    unittest.main()

"""
Unit Tests for Unicode-Aware and Multilingual Feature Builder
Tests Cyrillic detection, Spanish inverted punctuation, French/German quotation marks,
universal bullet markers, and cross-lingual case continuation.
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.features.feature_builder import (
    extract_boundary_features,
    extract_batch_features,
    FEATURE_NAMES,
    UNIVERSAL_BULLET_REGEX
)


class TestUnicodeFeatureBuilder(unittest.TestCase):

    def test_feature_vector_dimension(self):
        self.assertEqual(len(FEATURE_NAMES), 30)

    def test_spanish_inverted_punctuation(self):
        boundary = {
            'page_a': 0, 'page_b': 0,
            'bbox_a': [100, 100, 300, 115],
            'bbox_b': [100, 130, 300, 145],
            'text_a': 'Este es el final de la sección.',
            'text_b': '¿Cómo funciona el sistema educativo?'
        }
        vec = extract_boundary_features(boundary)
        # ends_with_terminal_punct should be 1.0
        self.assertEqual(vec[FEATURE_NAMES.index("ends_with_terminal_punct")], 1.0)
        # starts_with_inverted_punct should be 1.0
        self.assertEqual(vec[FEATURE_NAMES.index("starts_with_inverted_punct")], 1.0)
        # starts_with_uppercase should be 1.0 ('C' in '¿Cómo')
        self.assertEqual(vec[FEATURE_NAMES.index("starts_with_uppercase")], 1.0)
        self.assertEqual(vec[FEATURE_NAMES.index("starts_with_lowercase")], 0.0)

    def test_cyrillic_script_detection(self):
        # Russian / Ukrainian boundary
        boundary = {
            'page_a': 0, 'page_b': 0,
            'bbox_a': [100, 100, 300, 115],
            'bbox_b': [100, 120, 300, 135],
            'text_a': 'Введение в образовательную программу,',
            'text_b': 'которая предназначена для студентов.'
        }
        vec = extract_boundary_features(boundary)
        # is_cyrillic should be 1.0
        self.assertEqual(vec[FEATURE_NAMES.index("is_cyrillic")], 1.0)
        # ends_with_soft_punct (',') should be 1.0
        self.assertEqual(vec[FEATURE_NAMES.index("ends_with_soft_punct")], 1.0)
        # starts_with_lowercase ('к' in 'которая') should be 1.0
        self.assertEqual(vec[FEATURE_NAMES.index("starts_with_lowercase")], 1.0)
        self.assertEqual(vec[FEATURE_NAMES.index("starts_with_uppercase")], 0.0)

    def test_french_guillemets_and_punctuation(self):
        boundary = {
            'page_a': 0, 'page_b': 0,
            'bbox_a': [100, 100, 300, 115],
            'bbox_b': [100, 120, 300, 135],
            'text_a': 'Selon le rapport « Éducation »',
            'text_b': 'les élèves ont progressé.'
        }
        vec = extract_boundary_features(boundary)
        self.assertEqual(vec[FEATURE_NAMES.index("ends_with_quote_or_paren")], 1.0)
        self.assertEqual(vec[FEATURE_NAMES.index("starts_with_lowercase")], 1.0)

    def test_german_quotes_and_capitalized_continuation(self):
        boundary = {
            'page_a': 0, 'page_b': 0,
            'bbox_a': [100, 100, 300, 115],
            'bbox_b': [100, 120, 300, 135],
            'text_a': 'Das Konzept der „Bildung',
            'text_b': 'ist von zentraler Bedeutung.'
        }
        vec = extract_boundary_features(boundary)
        self.assertEqual(vec[FEATURE_NAMES.index("starts_with_lowercase")], 1.0)

    def test_universal_bullet_markers(self):
        test_markers = ['•', '-', '*', '1.', '2)', 'IV.', 'а)', 'б)', 'в.']
        for marker in test_markers:
            self.assertTrue(
                bool(UNIVERSAL_BULLET_REGEX.match(marker)),
                f"Marker '{marker}' should be recognized as a bullet"
            )

        non_markers = ['Hello world', 'Palavra', 'Образование']
        for text in non_markers:
            self.assertFalse(
                bool(UNIVERSAL_BULLET_REGEX.match(text)),
                f"Text '{text}' should NOT be recognized as a bullet"
            )


if __name__ == '__main__':
    unittest.main()

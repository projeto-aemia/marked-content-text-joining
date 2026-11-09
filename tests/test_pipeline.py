"""
End-to-End Pipeline Unit and Integration Tests
Verifies DOCX extraction, PDF extraction, alignment, feature building, and inference.
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.extraction.docx_extractor import extract_docx_paragraphs
from src.extraction.pdf_mc_extractor import extract_pdf_pieces
from src.extraction.aligner import align_pdf_docx
from src.features.feature_builder import extract_boundary_features, extract_batch_features, FEATURE_NAMES
from src.models.gbdt_model import GBDTBoundaryClassifier
from src.models.cascade_model import CascadeBoundaryClassifier


class TestTextJoiningPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.sample_docx = r"f:\docx-corpus-educational-education\005ae28b97352465b999ba0b5fe2d4792063e33e9e6d44c5c68ffdec3bbd0535.docx"
        cls.sample_pdf = r"f:\docx-corpus-educational-education\PDFs\005ae28b97352465b999ba0b5fe2d4792063e33e9e6d44c5c68ffdec3bbd0535.pdf"
        cls.has_sample_files = os.path.exists(cls.sample_docx) and os.path.exists(cls.sample_pdf)

    def test_docx_extractor(self):
        if not self.has_sample_files:
            self.skipTest("Sample files not found on disk")
        paras = extract_docx_paragraphs(self.sample_docx)
        self.assertGreater(len(paras), 50)
        self.assertIn("Perspectivas", paras[0]['text'])

    def test_pdf_extractor(self):
        if not self.has_sample_files:
            self.skipTest("Sample files not found on disk")
        pieces = extract_pdf_pieces(self.sample_pdf)
        self.assertGreater(len(pieces), 100)
        p0 = pieces[0]
        self.assertIn('bbox', p0)
        self.assertEqual(len(p0['bbox']), 4)

    def test_aligner_and_boundaries(self):
        if not self.has_sample_files:
            self.skipTest("Sample files not found on disk")
        boundaries = align_pdf_docx(self.sample_docx, self.sample_pdf)
        self.assertGreater(len(boundaries), 500)
        labels = [b['label'] for b in boundaries]
        self.assertIn(1, labels) # At least some joins
        self.assertIn(0, labels) # At least some splits

    def test_feature_extractor(self):
        dummy_boundary = {
            'page_a': 0, 'page_b': 0,
            'bbox_a': [100.0, 200.0, 400.0, 212.0],
            'bbox_b': [100.0, 214.0, 350.0, 226.0],
            'font_size_a': 12.0, 'font_size_b': 12.0,
            'font_a': 'Arial', 'font_b': 'Arial',
            'is_bold_a': False, 'is_bold_b': False,
            'text_a': 'Esta é a primeira linha da sentença,',
            'text_b': 'e esta é a continuação do parágrafo.'
        }
        vec = extract_boundary_features(dummy_boundary)
        self.assertEqual(len(vec), 30)
        self.assertEqual(len(vec), len(FEATURE_NAMES))
        # dy should be 2.0 (214.0 - 212.0)
        self.assertAlmostEqual(vec[1], 2.0, places=1)
        # starts_with_lowercase should be 1.0
        self.assertEqual(vec[22], 1.0)

    def test_gbdt_and_cascade_models(self):
        gbdt_path = os.path.join(project_root, "saved_models", "gbdt_model.pkl")
        if not os.path.exists(gbdt_path):
            self.skipTest("GBDT model not yet saved")
            
        clf = GBDTBoundaryClassifier.load(gbdt_path)
        
        # Test on a real known boundary from test dataset
        import json
        with open(os.path.join(project_root, "data", "test_boundaries.json"), "r", encoding="utf-8") as f:
            test_data = json.load(f)
            
        # Sample 4 is a clear body line continuation (True=1)
        # Sample 2 is a heading transition into 'Resumo:' (True=0)
        body_join_sample = test_data[4]
        body_split_sample = test_data[2]
        
        X_join, _ = extract_batch_features([body_join_sample])
        X_split, _ = extract_batch_features([body_split_sample])
        
        prob_join = clf.predict_proba(X_join)[0, 1]
        prob_split = clf.predict_proba(X_split)[0, 1]
        
        self.assertGreater(prob_join, 0.9)
        self.assertLess(prob_split, 0.1)
        
        cascade = CascadeBoundaryClassifier(clf)
        preds, probs, stats = cascade.predict_with_routing([body_join_sample, body_split_sample])
        self.assertEqual(len(preds), 2)
        self.assertEqual(preds[0], 1)
        self.assertEqual(preds[1], 0)


if __name__ == '__main__':
    unittest.main()

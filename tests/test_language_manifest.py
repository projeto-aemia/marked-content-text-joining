"""
Unit tests for language manifest generation and CLI filtering by document types/topics.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.extraction.language_manifest import (
    normalize_filter_list,
    generate_language_manifests,
    DEFAULT_TYPES,
    DEFAULT_TOPICS
)
from main import build_parser


class TestLanguageManifestFiltering(unittest.TestCase):

    def test_normalize_filter_list_defaults(self):
        # When unspecified (None), default is returned
        self.assertEqual(normalize_filter_list(None, default=DEFAULT_TYPES), ['educational'])
        self.assertEqual(normalize_filter_list(None, default=DEFAULT_TOPICS), ['education'])

        # When match_all is True, returns None regardless of default
        self.assertIsNone(normalize_filter_list(None, default=DEFAULT_TYPES, match_all=True))
        self.assertIsNone(normalize_filter_list(['educational'], default=DEFAULT_TYPES, match_all=True))

    def test_normalize_filter_list_custom_values(self):
        # Single string
        self.assertEqual(normalize_filter_list("educational"), ['educational'])
        self.assertEqual(normalize_filter_list("Educational"), ['educational'])

        # Comma-separated string
        self.assertEqual(normalize_filter_list("educational, government"), ['educational', 'government'])
        self.assertEqual(normalize_filter_list("education, science, research"), ['education', 'science', 'research'])

        # List of strings
        self.assertEqual(normalize_filter_list(["educational", "academic"]), ['educational', 'academic'])

        # List with comma-separated elements
        self.assertEqual(normalize_filter_list(["educational, academic", "legal"]), ['educational', 'academic', 'legal'])

    def test_normalize_filter_list_match_all_keywords(self):
        # 'all', '*', 'any' should disable filtering
        self.assertIsNone(normalize_filter_list("all"))
        self.assertIsNone(normalize_filter_list(["all"]))
        self.assertIsNone(normalize_filter_list(["*"]))
        self.assertIsNone(normalize_filter_list("any"))
        self.assertIsNone(normalize_filter_list([]))
        self.assertIsNone(normalize_filter_list(["   "]))

    @patch('src.extraction.language_manifest.datasets')
    @patch('os.listdir')
    @patch('os.path.exists')
    def test_generate_language_manifests_filtering(self, mock_exists, mock_listdir, mock_datasets):
        # Mock disk files
        mock_listdir.return_value = ['doc1.pdf', 'doc2.pdf', 'doc3.pdf', 'doc4.pdf']
        mock_exists.return_value = True

        mock_rows = [
            {'id': 'doc1', 'filename': 'doc1.docx', 'language': 'en', 'type': 'educational', 'topic': 'education'},
            {'id': 'doc2', 'filename': 'doc2.docx', 'language': 'en', 'type': 'scientific', 'topic': 'science'},
            {'id': 'doc3', 'filename': 'doc3.docx', 'language': 'pt', 'type': 'educational', 'topic': 'science'},
            {'id': 'doc4', 'filename': 'doc4.docx', 'language': 'pt', 'type': 'educational', 'topic': 'education'},
        ]

        class MockDataset:
            def __init__(self, data):
                self._data = data

            def filter(self, fn):
                return MockDataset([r for r in self._data if fn(r)])

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

        mock_ds = MockDataset(mock_rows)
        mock_datasets.load_dataset.return_value = mock_ds

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Default: unspecified types and topics -> educational + education
            summary = generate_language_manifests(
                docx_dir=tmpdir,
                pdf_dir=tmpdir,
                output_dir=tmpdir,
                target_languages=['en', 'pt']
            )
            # Only doc1 (en) and doc4 (pt) match
            self.assertEqual(summary['en'], 1)
            self.assertEqual(summary['pt'], 1)
            self.assertEqual(summary['_total_target_documents'], 2)

            # 2. Custom types and topics: educational type, all topics
            summary_custom = generate_language_manifests(
                docx_dir=tmpdir,
                pdf_dir=tmpdir,
                output_dir=tmpdir,
                target_languages=['en', 'pt'],
                types=['educational'],
                topics=['all']
            )
            # doc1 (en, educational, education), doc3 (pt, educational, science), doc4 (pt, educational, education)
            self.assertEqual(summary_custom['en'], 1)
            self.assertEqual(summary_custom['pt'], 2)
            self.assertEqual(summary_custom['_total_target_documents'], 3)

            # 3. All types and topics (entire corpus)
            summary_all = generate_language_manifests(
                docx_dir=tmpdir,
                pdf_dir=tmpdir,
                output_dir=tmpdir,
                target_languages=['en', 'pt'],
                all_types=True,
                all_topics=True
            )
            # All 4 docs match
            self.assertEqual(summary_all['en'], 2)
            self.assertEqual(summary_all['pt'], 2)
            self.assertEqual(summary_all['_total_target_documents'], 4)


class TestCliManifestParsing(unittest.TestCase):

    def setUp(self):
        self.parser = build_parser()

    def test_default_manifest_args(self):
        args = self.parser.parse_args(["manifest"])
        self.assertIsNone(args.types)
        self.assertIsNone(args.topics)
        self.assertFalse(args.all_types)
        self.assertFalse(args.all_topics)
        self.assertFalse(args.all)

    def test_space_separated_types_and_topics(self):
        args = self.parser.parse_args([
            "manifest",
            "--types", "educational", "government",
            "--topics", "education", "politics"
        ])
        self.assertEqual(args.types, ["educational", "government"])
        self.assertEqual(args.topics, ["education", "politics"])

    def test_comma_separated_types_and_topics(self):
        args = self.parser.parse_args([
            "manifest",
            "--types", "educational,government",
            "--topics", "education,politics"
        ])
        self.assertEqual(args.types, ["educational,government"])
        self.assertEqual(args.topics, ["education,politics"])
        self.assertEqual(normalize_filter_list(args.types), ["educational", "government"])
        self.assertEqual(normalize_filter_list(args.topics), ["education", "politics"])

    def test_all_flags(self):
        args = self.parser.parse_args(["manifest", "--all"])
        self.assertTrue(args.all)

        args_separate = self.parser.parse_args(["manifest", "--all-types", "--all-topics"])
        self.assertTrue(args_separate.all_types)
        self.assertTrue(args_separate.all_topics)

    def test_all_keyword_values(self):
        args = self.parser.parse_args(["manifest", "--types", "all", "--topics", "all"])
        self.assertEqual(args.types, ["all"])
        self.assertEqual(args.topics, ["all"])
        self.assertIsNone(normalize_filter_list(args.types, default=DEFAULT_TYPES))
        self.assertIsNone(normalize_filter_list(args.topics, default=DEFAULT_TOPICS))


if __name__ == '__main__':
    unittest.main()

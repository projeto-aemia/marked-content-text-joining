"""
Language Manifest Generator for DOCX-Corpus
Discovers valid PDF-DOCX document pairs for target languages and generates
per-language manifest files and a summary report. Supports filtering by
document types and topics, defaulting to 'educational' and 'education'.
"""

import os
import sys
import json
from collections import defaultdict
from typing import Dict, List, Optional, Union, Any

try:
    import datasets
except ImportError:
    datasets = None

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TARGET_LANGUAGES = ['en', 'ru', 'es', 'tr', 'fr', 'cs', 'pt', 'de', 'pl', 'uk', 'id']
DEFAULT_TYPES = ['educational']
DEFAULT_TOPICS = ['education']


def normalize_filter_list(
    values: Optional[Union[List[str], str]],
    default: Optional[List[str]] = None,
    match_all: bool = False
) -> Optional[List[str]]:
    """
    Normalizes a filter argument (string, comma-separated string, or list of strings).
    Returns None if all values should match (i.e. no filtering).
    Returns a list of lowercase string filters otherwise.
    """
    if match_all:
        return None
    if values is None:
        if default is None:
            return None
        values = default
    if isinstance(values, str):
        values = [v.strip() for v in values.split(",") if v.strip()]
    elif isinstance(values, (list, tuple)):
        items = []
        for v in values:
            for part in str(v).split(","):
                part = part.strip()
                if part:
                    items.append(part)
        values = items

    cleaned = [str(v).strip().lower() for v in values if str(v).strip()]
    if not cleaned or any(v in ('all', '*', 'any') for v in cleaned):
        return None  # No filtering; match all
    return cleaned


def generate_language_manifests(
    docx_dir: str = r"f:\docx-corpus-educational-education",
    pdf_dir: str = r"f:\docx-corpus-educational-education\PDFs",
    output_dir: str = os.path.join(PROJECT_ROOT, "manifests"),
    target_languages: Optional[List[str]] = None,
    types: Optional[Union[List[str], str]] = None,
    topics: Optional[Union[List[str], str]] = None,
    all_types: bool = False,
    all_topics: bool = False
) -> Dict[str, int]:
    """
    Builds per-language manifests for valid document pairs (both .docx and .pdf present).
    Filters by document types and topics (defaults to 'educational' and 'education').
    Pass 'all' or all_types=True/all_topics=True to include the entire corpus.
    """
    if target_languages is None:
        target_languages = TARGET_LANGUAGES

    norm_types = normalize_filter_list(types, default=DEFAULT_TYPES, match_all=all_types)
    norm_topics = normalize_filter_list(topics, default=DEFAULT_TOPICS, match_all=all_topics)

    type_desc = ", ".join(norm_types) if norm_types is not None else "ALL"
    topic_desc = ", ".join(norm_topics) if norm_topics is not None else "ALL"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Scanning converted PDFs in {pdf_dir}...")
    converted_pdfs = set(
        f[:-4] for f in os.listdir(pdf_dir)
        if f.endswith('.pdf') and not f.startswith('_')
    )
    print(f"Found {len(converted_pdfs)} converted PDF files on disk.")

    print("Loading HuggingFace dataset metadata (superdoc-dev/docx-corpus)...")
    global datasets
    if datasets is None:
        try:
            import datasets
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required to generate language manifests from superdoc-dev/docx-corpus. "
                "Please install it using 'pip install datasets'."
            )
    ds = datasets.load_dataset('superdoc-dev/docx-corpus', split='train')

    if norm_types is not None or norm_topics is not None:
        def doc_filter(x):
            if norm_types is not None:
                val = x.get('type')
                if val is None or str(val).strip().lower() not in norm_types:
                    return False
            if norm_topics is not None:
                val = x.get('topic')
                if val is None or str(val).strip().lower() not in norm_topics:
                    return False
            return True

        print(f"Filtering dataset by types: [{type_desc}], topics: [{topic_desc}]...")
        filtered_ds = ds.filter(doc_filter)
        print(f"Total documents matching filter ({type_desc} + {topic_desc}): {len(filtered_ds)}")
    else:
        print("No type/topic filtering applied (using entire corpus).")
        filtered_ds = ds
        print(f"Total documents in corpus: {len(filtered_ds)}")

    by_lang = defaultdict(list)
    missing_docx = 0
    missing_pdf = 0

    for row in filtered_ds:
        h = row['id'] if 'id' in row and row['id'] else row['filename'].replace('.docx', '')
        lang = row['language']

        if lang not in target_languages:
            continue

        if h not in converted_pdfs:
            missing_pdf += 1
            continue

        docx_path = os.path.join(docx_dir, f"{h}.docx")
        if not os.path.exists(docx_path):
            missing_docx += 1
            continue

        by_lang[lang].append(h)

    print(f"Filtering complete. Missing PDFs: {missing_pdf}, Missing DOCX: {missing_docx}")

    summary = {}
    total_valid = 0

    for lang in target_languages:
        hashes = by_lang.get(lang, [])
        m_path = os.path.join(output_dir, f"{lang}_manifest.txt")
        with open(m_path, "w", encoding="utf-8") as f:
            for h in hashes:
                f.write(f"https://docxcorp.us/documents/{h}.docx\n")

        summary[lang] = len(hashes)
        total_valid += len(hashes)
        print(f"  [{lang:>2}] {len(hashes):>5} valid document pairs saved to {lang}_manifest.txt")

    summary["_total_target_documents"] = total_valid
    summary_path = os.path.join(output_dir, "manifest_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nManifest generation complete! Total valid pairs across target languages: {total_valid}")
    print(f"Summary written to {summary_path}")
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate language manifest files from superdoc-dev/docx-corpus with optional type/topic filtering"
    )
    parser.add_argument(
        "--docx-dir",
        default=r"f:\docx-corpus-educational-education",
        help="Path to source DOCX documents"
    )
    parser.add_argument(
        "--pdf-dir",
        default=r"f:\docx-corpus-educational-education\PDFs",
        help="Path to converted PDF documents"
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "manifests"),
        help="Output directory for manifests"
    )
    parser.add_argument(
        "--languages",
        type=str,
        default=None,
        help="Comma-separated language codes"
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=None,
        help="List of document types to filter (space or comma-separated, e.g. 'educational', 'all' for no filter). Default: educational"
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=None,
        help="List of document topics to filter (space or comma-separated, e.g. 'education', 'all' for no filter). Default: education"
    )
    parser.add_argument(
        "--all-types",
        action="store_true",
        help="Include all document types without filtering"
    )
    parser.add_argument(
        "--all-topics",
        action="store_true",
        help="Include all document topics without filtering"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include all document types and topics (entire corpus) without filtering"
    )
    args = parser.parse_args()
    langs = args.languages.split(",") if args.languages else TARGET_LANGUAGES
    generate_language_manifests(
        docx_dir=args.docx_dir,
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        target_languages=langs,
        types=args.types,
        topics=args.topics,
        all_types=args.all_types or args.all,
        all_topics=args.all_topics or args.all
    )

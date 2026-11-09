"""
Multilingual Dataset Builder for Marked Content Text Joining
Extracts aligned boundaries across 11 target languages and creates
leak-free document-level Train, Validation, and Test datasets (both per-language and unified global).
"""

import os
import sys
import json
import random
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.extraction.aligner import align_pdf_docx

TARGET_LANGUAGES = ['en', 'ru', 'es', 'tr', 'fr', 'cs', 'pt', 'de', 'pl', 'uk', 'id']


def build_multilingual_dataset(
    manifest_dir: str = os.path.join(PROJECT_ROOT, "manifests"),
    docx_dir: str = r"f:\docx-corpus-educational-education",
    pdf_dir: str = r"f:\docx-corpus-educational-education\PDFs",
    output_dir: str = os.path.join(PROJECT_ROOT, "data", "multilingual"),
    train_docs_per_lang: int = 60,
    val_docs_per_lang: int = 10,
    test_docs_per_lang: int = 15,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Builds document-level leak-free Train/Val/Test splits for all target languages.
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    summary: Dict[str, Any] = {
        "per_language": {},
        "global": {
            "total_train_boundaries": 0,
            "total_val_boundaries": 0,
            "total_test_boundaries": 0,
            "total_documents": 0
        }
    }

    all_train_boundaries: List[Dict[str, Any]] = []
    all_val_boundaries: List[Dict[str, Any]] = []
    all_test_boundaries: List[Dict[str, Any]] = []

    print("=" * 70)
    print("BUILDING MULTILINGUAL MARKED CONTENT DATASET")
    print(f"Target languages: {', '.join(TARGET_LANGUAGES)}")
    print(f"Per-language quotas: {train_docs_per_lang} Train / {val_docs_per_lang} Val / {test_docs_per_lang} Test docs")
    print("=" * 70)

    for lang in TARGET_LANGUAGES:
        manifest_file = os.path.join(manifest_dir, f"{lang}_manifest.txt")
        if not os.path.exists(manifest_file):
            print(f"Warning: Manifest for {lang} not found at {manifest_file}. Skipping.")
            continue

        with open(manifest_file, "r", encoding="utf-8") as f:
            hashes = [
                line.strip().split('/')[-1].replace('.docx', '')
                for line in f if line.strip()
            ]

        # Verify physical existence
        valid_hashes = []
        for h in hashes:
            d_path = os.path.join(docx_dir, f"{h}.docx")
            p_path = os.path.join(pdf_dir, f"{h}.pdf")
            if os.path.exists(d_path) and os.path.exists(p_path):
                valid_hashes.append(h)

        random.shuffle(valid_hashes)
        total_needed = train_docs_per_lang + val_docs_per_lang + test_docs_per_lang
        selected = valid_hashes[:total_needed]

        train_docs = selected[:train_docs_per_lang]
        val_docs = selected[train_docs_per_lang : train_docs_per_lang + val_docs_per_lang]
        test_docs = selected[train_docs_per_lang + val_docs_per_lang :]

        print(f"\n[{lang.upper()}] Processing {len(train_docs)} Train, {len(val_docs)} Val, {len(test_docs)} Test documents...")

        def process_doc_list(split_name: str, doc_list: List[str]) -> List[Dict[str, Any]]:
            boundaries = []
            for i, h in enumerate(doc_list):
                d_path = os.path.join(docx_dir, f"{h}.docx")
                p_path = os.path.join(pdf_dir, f"{h}.pdf")
                try:
                    b_list = align_pdf_docx(d_path, p_path)
                    # Tag with language and doc_id
                    for b in b_list:
                        b['lang'] = lang
                        b['doc_id'] = h
                    boundaries.extend(b_list)
                except Exception as e:
                    print(f"  [{lang} {split_name}] Error extracting {h[:8]}: {e}")
            print(f"  [{lang} {split_name}] Extracted {len(boundaries)} boundaries across {len(doc_list)} docs.")
            return boundaries

        lang_train = process_doc_list("TRAIN", train_docs)
        lang_val = process_doc_list("VAL", val_docs)
        lang_test = process_doc_list("TEST", test_docs)

        # Save individual language split files
        with open(os.path.join(output_dir, f"{lang}_train.json"), "w", encoding="utf-8") as f:
            json.dump(lang_train, f, ensure_ascii=False)
        with open(os.path.join(output_dir, f"{lang}_val.json"), "w", encoding="utf-8") as f:
            json.dump(lang_val, f, ensure_ascii=False)
        with open(os.path.join(output_dir, f"{lang}_test.json"), "w", encoding="utf-8") as f:
            json.dump(lang_test, f, ensure_ascii=False)

        summary["per_language"][lang] = {
            "train_docs": len(train_docs),
            "val_docs": len(val_docs),
            "test_docs": len(test_docs),
            "train_boundaries": len(lang_train),
            "val_boundaries": len(lang_val),
            "test_boundaries": len(lang_test),
            "train_join_ratio": float(sum(1 for b in lang_train if b['label'] == 1) / max(1, len(lang_train))),
            "test_join_ratio": float(sum(1 for b in lang_test if b['label'] == 1) / max(1, len(lang_test)))
        }

        all_train_boundaries.extend(lang_train)
        all_val_boundaries.extend(lang_val)
        all_test_boundaries.extend(lang_test)

    # Save Global unified splits
    print("\nSaving Global Unified Splits...")
    with open(os.path.join(output_dir, "global_train.json"), "w", encoding="utf-8") as f:
        json.dump(all_train_boundaries, f, ensure_ascii=False)
    with open(os.path.join(output_dir, "global_val.json"), "w", encoding="utf-8") as f:
        json.dump(all_val_boundaries, f, ensure_ascii=False)
    with open(os.path.join(output_dir, "global_test.json"), "w", encoding="utf-8") as f:
        json.dump(all_test_boundaries, f, ensure_ascii=False)

    summary["global"]["total_train_boundaries"] = len(all_train_boundaries)
    summary["global"]["total_val_boundaries"] = len(all_val_boundaries)
    summary["global"]["total_test_boundaries"] = len(all_test_boundaries)
    summary["global"]["total_documents"] = sum(
        v["train_docs"] + v["val_docs"] + v["test_docs"]
        for v in summary["per_language"].values()
    )

    summary_file = os.path.join(output_dir, "multilingual_dataset_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("MULTILINGUAL DATASET SUMMARY:")
    print(f"Total documents processed: {summary['global']['total_documents']}")
    print(f"Total train boundaries:   {summary['global']['total_train_boundaries']}")
    print(f"Total val boundaries:     {summary['global']['total_val_boundaries']}")
    print(f"Total test boundaries:    {summary['global']['total_test_boundaries']}")
    print("=" * 70)
    return summary


if __name__ == '__main__':
    build_multilingual_dataset()

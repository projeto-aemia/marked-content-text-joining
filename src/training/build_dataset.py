"""
Dataset Builder for Marked Content Text Joining
Extracts aligned boundaries across multiple Portuguese PDF/DOCX pairs
and creates leak-free document-level Train, Validation, and Test datasets.
"""

import os
import sys
import json
import random
from typing import List, Dict, Any

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.extraction.aligner import align_pdf_docx
from src.features.feature_builder import extract_batch_features


def build_dataset(
    manifest_path: str = "docx-corpus-educational-education-pt-manifest.txt",
    docx_dir: str = r"f:\docx-corpus-educational-education",
    pdf_dir: str = r"f:\docx-corpus-educational-education\PDFs",
    output_dir: str = "data",
    num_docs: int = 30,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Builds train/val/test splits across `num_docs` document pairs.
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        hashes = [line.strip().split('/')[-1].replace('.docx', '') for line in f if line.strip()]
        
    valid_hashes = []
    for h in hashes:
        d_path = os.path.join(docx_dir, f"{h}.docx")
        p_path = os.path.join(pdf_dir, f"{h}.pdf")
        if os.path.exists(d_path) and os.path.exists(p_path):
            valid_hashes.append(h)
            
    print(f"Total available document pairs: {len(valid_hashes)}")
    selected_hashes = valid_hashes[:num_docs]
    random.shuffle(selected_hashes)
    
    # 80% train, 10% val, 10% test by document
    n_total = len(selected_hashes)
    n_train = max(1, int(n_total * 0.8))
    n_val = max(1, int(n_total * 0.1))
    
    train_docs = selected_hashes[:n_train]
    val_docs = selected_hashes[n_train : n_train + n_val]
    test_docs = selected_hashes[n_train + n_val :]
    
    print(f"Split distribution: {len(train_docs)} Train docs, {len(val_docs)} Val docs, {len(test_docs)} Test docs")
    
    def process_split(split_name: str, doc_list: List[str]) -> List[Dict[str, Any]]:
        split_boundaries = []
        for i, h in enumerate(doc_list):
            d_path = os.path.join(docx_dir, f"{h}.docx")
            p_path = os.path.join(pdf_dir, f"{h}.pdf")
            try:
                b_list = align_pdf_docx(d_path, p_path)
                split_boundaries.extend(b_list)
                print(f"[{split_name} {i+1}/{len(doc_list)}] Doc {h[:8]}: {len(b_list)} boundaries extracted.")
            except Exception as e:
                print(f"Error processing doc {h}: {e}")
        return split_boundaries

    train_data = process_split("TRAIN", train_docs)
    val_data = process_split("VAL", val_docs)
    test_data = process_split("TEST", test_docs)
    
    # Save JSON files
    with open(os.path.join(output_dir, "train_boundaries.json"), "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
        
    with open(os.path.join(output_dir, "val_boundaries.json"), "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
        
    with open(os.path.join(output_dir, "test_boundaries.json"), "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
        
    summary = {
        "num_docs_total": n_total,
        "train_docs": len(train_docs),
        "val_docs": len(val_docs),
        "test_docs": len(test_docs),
        "train_boundaries": len(train_data),
        "val_boundaries": len(val_data),
        "test_boundaries": len(test_data),
        "train_join_ratio": float(sum(1 for b in train_data if b['label'] == 1) / max(1, len(train_data))),
        "val_join_ratio": float(sum(1 for b in val_data if b['label'] == 1) / max(1, len(val_data))),
        "test_join_ratio": float(sum(1 for b in test_data if b['label'] == 1) / max(1, len(test_data)))
    }
    
    with open(os.path.join(output_dir, "dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("\nDataset generation complete!")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    # Build dataset using 15 documents for initial development/benchmarking
    build_dataset(num_docs=15)

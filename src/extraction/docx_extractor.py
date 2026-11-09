"""
DOCX Paragraph Extractor
Extracts author-intended paragraphs, styles, and list markers from DOCX files
to serve as ground truth for text joining.
"""

import os
from typing import List, Dict, Any
import docx
from docx.document import Document as DocxDocument


def extract_docx_paragraphs(docx_path: str) -> List[Dict[str, Any]]:
    """
    Extracts ordered non-empty paragraphs from a DOCX file.
    
    Returns a list of dicts:
    [
        {
            'para_id': int,
            'text': str,
            'style': str,
            'is_heading': bool,
            'is_list': bool,
            'run_count': int
        },
        ...
    ]
    """
    if not os.path.exists(docx_path):
        raise FileNotFoundError(f"DOCX file not found: {docx_path}")
        
    doc = docx.Document(docx_path)
    extracted = []
    
    para_idx = 0
    for p in doc.paragraphs:
        raw_text = p.text.strip()
        if not raw_text:
            continue
            
        style_name = p.style.name if p.style else ""
        is_heading = "heading" in style_name.lower() or "título" in style_name.lower() or "title" in style_name.lower()
        is_list = "list" in style_name.lower() or "bullet" in style_name.lower() or "item" in style_name.lower()
        
        # Check if text itself starts with a bullet marker or numbered list
        if raw_text.startswith(('•', '-', '*', '–')) or any(raw_text.startswith(f"{i}.") for i in range(1, 100)):
            is_list = True

        extracted.append({
            'para_id': para_idx,
            'text': raw_text,
            'style': style_name,
            'is_heading': is_heading,
            'is_list': is_list,
            'run_count': len(p.runs)
        })
        para_idx += 1
        
    return extracted


if __name__ == '__main__':
    import sys
    test_path = sys.argv[1] if len(sys.argv) > 1 else r"f:\docx-corpus-educational-education\005ae28b97352465b999ba0b5fe2d4792063e33e9e6d44c5c68ffdec3bbd0535.docx"
    paras = extract_docx_paragraphs(test_path)
    print(f"Extracted {len(paras)} paragraphs from {os.path.basename(test_path)}:")
    for p in paras[:5]:
        print(f"[{p['para_id']}] ({p['style']}) {repr(p['text'][:60])}")

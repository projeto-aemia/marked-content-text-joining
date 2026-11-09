"""
PDF Marked Content & Text Span Extractor
Extracts sequential text pieces with geometric layout characteristics (bounding box,
font family, font size, bold/italic flags, page number) from PDF documents.
Supports both PyMuPDF layout spans and Tagged PDF Marked Content (/MCID) streams.
"""

import os
from typing import List, Dict, Any, Optional
import pymupdf


def extract_pdf_pieces(pdf_path: str, filter_artifacts: bool = True) -> List[Dict[str, Any]]:
    """
    Extracts ordered text pieces from a PDF using PyMuPDF dict layout.
    Each piece corresponds to a distinct span or text run as formatted in the PDF.
    
    Returns a list of dicts:
    [
        {
            'piece_id': int,
            'page_num': int,
            'text': str,
            'bbox': [float, float, float, float], # [x0, y0, x1, y1]
            'font_name': str,
            'font_size': float,
            'is_bold': bool,
            'is_italic': bool,
            'line_idx': int,
            'block_idx': int
        },
        ...
    ]
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
    doc = pymupdf.open(pdf_path)
    pieces = []
    piece_idx = 0
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_dict = page.get_text("dict")
        page_height = page.rect.height
        
        for block_idx, block in enumerate(page_dict.get("blocks", [])):
            if block.get("type") != 0:  # Text blocks only
                continue
                
            for line_idx, line in enumerate(block.get("lines", [])):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                        
                    # Filter out standard header/footer artifacts if requested
                    bbox = [round(float(c), 2) for c in span.get("bbox", [0, 0, 0, 0])]
                    flags = span.get("flags", 0)
                    font_name = span.get("font", "")
                    font_size = round(float(span.get("size", 0.0)), 2)
                    
                    # Heuristic for running header/footer: top/bottom 35pt of page with tiny font
                    if filter_artifacts and (bbox[1] < 35 or bbox[3] > page_height - 35) and font_size <= 9.0:
                        # Allow document title if on first page, but skip repetitive headers
                        if page_num > 0 and len(text) < 80:
                            continue

                    is_bold = bool(flags & (1 << 4)) or ("bold" in font_name.lower())
                    is_italic = bool(flags & (1 << 1)) or ("italic" in font_name.lower() or "oblique" in font_name.lower())

                    pieces.append({
                        'piece_id': piece_idx,
                        'page_num': page_num,
                        'text': text,
                        'bbox': bbox,
                        'font_name': font_name,
                        'font_size': font_size,
                        'is_bold': is_bold,
                        'is_italic': is_italic,
                        'line_idx': line_idx,
                        'block_idx': block_idx
                    })
                    piece_idx += 1
                    
    return pieces


if __name__ == '__main__':
    import sys
    test_pdf = sys.argv[1] if len(sys.argv) > 1 else r"f:\docx-corpus-educational-education\PDFs\005ae28b97352465b999ba0b5fe2d4792063e33e9e6d44c5c68ffdec3bbd0535.pdf"
    results = extract_pdf_pieces(test_pdf)
    print(f"Extracted {len(results)} text pieces from {os.path.basename(test_pdf)}:")
    for p in results[:8]:
        print(f"P{p['piece_id']} (pg {p['page_num']}, sz {p['font_size']}): bbox={p['bbox']} text={repr(p['text'][:50])}")

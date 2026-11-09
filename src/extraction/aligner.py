"""
DOCX <-> PDF Sequence Aligner
Aligns PDF Marked Content / layout pieces with author-intended DOCX paragraphs
to automatically produce gold-standard labeled boundaries (1 = JOIN, 0 = SPLIT).
"""

import os
import sys
import re
from typing import List, Dict, Any, Tuple, Optional

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.extraction.docx_extractor import extract_docx_paragraphs
from src.extraction.pdf_mc_extractor import extract_pdf_pieces


def clean_text_for_search(text: str) -> str:
    """Normalizes whitespace while preserving word characters and punctuation."""
    return re.sub(r'\s+', ' ', text).strip()


def align_pdf_docx(docx_path: str, pdf_path: str) -> List[Dict[str, Any]]:
    """
    Aligns PDF pieces with DOCX paragraphs and returns labeled boundary pairs.
    
    Each returned boundary dictionary contains:
    {
        'doc_id': str,
        'boundary_id': str,
        'piece_a_id': int,
        'piece_b_id': int,
        'text_a': str,
        'text_b': str,
        'context_pre': str,
        'context_post': str,
        'page_a': int,
        'page_b': int,
        'bbox_a': [float, float, float, float],
        'bbox_b': [float, float, float, float],
        'font_a': str,
        'font_b': str,
        'font_size_a': float,
        'font_size_b': float,
        'is_bold_a': bool,
        'is_bold_b': bool,
        'is_italic_a': bool,
        'is_italic_b': bool,
        'label': int, # 1 = JOIN, 0 = SPLIT
        'para_id_a': int,
        'para_id_b': int
    }
    """
    doc_id = os.path.splitext(os.path.basename(docx_path))[0]
    paras = extract_docx_paragraphs(docx_path)
    pieces = extract_pdf_pieces(pdf_path)
    
    if not paras or not pieces:
        return []
        
    # Build continuous DOCX character coordinate map
    docx_char_map: List[Tuple[int, int, int, str]] = []
    full_docx_text = ""
    for p in paras:
        start_pos = len(full_docx_text)
        clean_p = p['text'].strip()
        full_docx_text += clean_p + "\n"
        end_pos = len(full_docx_text)
        docx_char_map.append((start_pos, end_pos, p['para_id'], clean_p))
        
    # Sequential forward alignment of PDF pieces into DOCX text
    last_found_pos = 0
    piece_para_map: List[Optional[int]] = []
    
    for pi, piece in enumerate(pieces):
        raw_txt = piece['text']
        txt = clean_text_for_search(raw_txt)
        
        # If very short (e.g. single bullet '•', footnote '1', single letter),
        # inherit paragraph of surrounding context if within valid range
        if len(txt) < 3:
            if piece_para_map and piece_para_map[-1] is not None:
                piece_para_map.append(piece_para_map[-1])
            else:
                piece_para_map.append(None)
            continue
            
        # Search forward in a moving window
        search_start = max(0, last_found_pos - 200)
        search_window = full_docx_text[search_start : last_found_pos + 6000]
        idx = search_window.find(txt)
        
        found_pos = -1
        if idx != -1:
            found_pos = search_start + idx
            last_found_pos = found_pos + len(txt)
        else:
            # Fallback prefix matching (for hyphenated or slightly wrapped lines)
            prefix = txt[:min(25, len(txt))]
            idx_prefix = search_window.find(prefix)
            if idx_prefix != -1:
                found_pos = search_start + idx_prefix
                last_found_pos = found_pos + len(prefix)
            else:
                # Suffix fallback
                suffix = txt[-min(25, len(txt)):]
                idx_suffix = search_window.find(suffix)
                if idx_suffix != -1:
                    found_pos = search_start + idx_suffix
                    last_found_pos = found_pos + len(suffix)
                    
        if found_pos != -1:
            # Determine which paragraph contains found_pos
            assigned_para_id = None
            for s_pos, e_pos, pid, _ in docx_char_map:
                if s_pos <= found_pos < e_pos:
                    assigned_para_id = pid
                    break
            piece_para_map.append(assigned_para_id)
        else:
            piece_para_map.append(None)
            
    # Generate boundaries between consecutive pieces
    boundaries: List[Dict[str, Any]] = []
    
    for i in range(len(pieces) - 1):
        para_a = piece_para_map[i]
        para_b = piece_para_map[i + 1]
        
        # Both pieces must be reliably matched to document paragraphs
        if para_a is None or para_b is None:
            continue
            
        piece_a = pieces[i]
        piece_b = pieces[i + 1]
        
        # Determine ground truth label
        # 1 = JOIN (same paragraph), 0 = SPLIT (different paragraph)
        label = 1 if (para_a == para_b) else 0
        
        context_pre = pieces[i - 1]['text'] if i > 0 else ""
        context_post = pieces[i + 2]['text'] if i + 2 < len(pieces) else ""
        
        boundary = {
            'doc_id': doc_id,
            'boundary_id': f"{doc_id}_p{piece_a['page_num']}_{piece_a['piece_id']}_{piece_b['piece_id']}",
            'piece_a_id': piece_a['piece_id'],
            'piece_b_id': piece_b['piece_id'],
            'text_a': piece_a['text'],
            'text_b': piece_b['text'],
            'context_pre': context_pre,
            'context_post': context_post,
            'page_a': piece_a['page_num'],
            'page_b': piece_b['page_num'],
            'bbox_a': piece_a['bbox'],
            'bbox_b': piece_b['bbox'],
            'font_a': piece_a['font_name'],
            'font_b': piece_b['font_name'],
            'font_size_a': piece_a['font_size'],
            'font_size_b': piece_b['font_size'],
            'is_bold_a': piece_a['is_bold'],
            'is_bold_b': piece_b['is_bold'],
            'is_italic_a': piece_a['is_italic'],
            'is_italic_b': piece_b['is_italic'],
            'para_id_a': para_a,
            'para_id_b': para_b,
            'label': label
        }
        boundaries.append(boundary)
        
    return boundaries


if __name__ == '__main__':
    import sys
    test_docx = sys.argv[1] if len(sys.argv) > 1 else r"f:\docx-corpus-educational-education\005ae28b97352465b999ba0b5fe2d4792063e33e9e6d44c5c68ffdec3bbd0535.docx"
    test_pdf = sys.argv[2] if len(sys.argv) > 2 else r"f:\docx-corpus-educational-education\PDFs\005ae28b97352465b999ba0b5fe2d4792063e33e9e6d44c5c68ffdec3bbd0535.pdf"
    
    b_list = align_pdf_docx(test_docx, test_pdf)
    joins = sum(1 for b in b_list if b['label'] == 1)
    splits = sum(1 for b in b_list if b['label'] == 0)
    print(f"Total labeled boundaries extracted: {len(b_list)} (JOIN: {joins}, SPLIT: {splits})")
    print("Sample boundary transitions:")
    for b in b_list[:6]:
        decision = "JOIN" if b['label'] == 1 else "SPLIT"
        print(f"[{decision}] {repr(b['text_a'][:35])}  --->  {repr(b['text_b'][:35])}")

"""
PDF Text Extraction for CARDIA Documentation.

Hybrid approach:
1. Try pdfplumber (fast, handles extractable text + columns)
2. Fallback to OCR for image-based pages
3. Use VLM for complex pages (tables, diagrams) when OCR quality is poor
"""

import os
import pdfplumber
from typing import Tuple, List, Dict, Optional
from pdf2image import convert_from_path
import pytesseract
from pytesseract import Output

try:
    from preprocessing.ocr_quality import should_use_vlm
    from preprocessing.vlm_utils import extract_page_with_vlm
except ImportError:
    from ocr_quality import should_use_vlm
    from vlm_utils import extract_page_with_vlm


def _check_ocr_quality_and_decide(ocr_text: str, ocr_confidence: Optional[float], 
                                   pdf_path: str, page_num: int,
                                   use_vlm_fallback: bool, confidence_threshold: float,
                                   verbose: bool = False) -> Tuple[str, str]:
    """
    Check OCR quality and decide whether to use OCR or fallback to VLM.
    
    Returns:
        (text, method): Either OCR text/method or VLM text/method
    """
    # Step 1: Check OCR confidence
    if use_vlm_fallback and ocr_confidence is not None and ocr_confidence < confidence_threshold:
        if verbose:
            print(f"  -> Low OCR confidence ({ocr_confidence:.1f}), using VLM...")
        vlm_text = extract_page_with_vlm(pdf_path, page_num)
        return vlm_text, "vlm"
    
    # Step 2: Check for complex tables
    if use_vlm_fallback and should_use_vlm(ocr_text):
        if verbose:
            print(f"  -> Complex table detected, using VLM...")
        vlm_text = extract_page_with_vlm(pdf_path, page_num)
        return vlm_text, "vlm"
    
    return ocr_text, "ocr"


def _run_ocr(pdf_path: str, page_num: int) -> Tuple[Optional[str], Optional[float]]:
    """
    Run OCR on a specific page.
    
    Returns:
        (text, avg_confidence): OCR text and average confidence score (0-100)
                               Returns (None, None) if OCR fails
    """
    images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
    if not images:
        return None, None
    
    # Get detailed OCR data including confidence scores
    ocr_data = pytesseract.image_to_data(images[0], output_type=Output.DICT)
    
    # Calculate average confidence (excluding -1 values which indicate no text)
    confidences = [conf for conf in ocr_data['conf'] if conf != -1]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # Reconstruct text from ocr_data to avoid running OCR twice
    # Group by block -> paragraph -> line for proper text structure
    text_structure = {}
    for i, word in enumerate(ocr_data['text']):
        if word.strip():  # Non-empty word
            block = ocr_data['block_num'][i]
            par = ocr_data['par_num'][i]
            line = ocr_data['line_num'][i]
            
            if block not in text_structure:
                text_structure[block] = {}
            if par not in text_structure[block]:
                text_structure[block][par] = {}
            if line not in text_structure[block][par]:
                text_structure[block][par][line] = []
            
            text_structure[block][par][line].append(word)
    
    # Join words -> lines -> paragraphs -> blocks
    text_parts = []
    for block in sorted(text_structure.keys()):
        for par in sorted(text_structure[block].keys()):
            par_lines = []
            for line in sorted(text_structure[block][par].keys()):
                par_lines.append(' '.join(text_structure[block][par][line]))
            text_parts.append('\n'.join(par_lines))
    
    text = '\n\n'.join(text_parts) if text_parts else ""
    
    return text, avg_confidence


def extract_page_text(pdf, page_num: int, pdf_path: str,
                     min_text_threshold: int = 100,
                     vertical_coverage_threshold: float = 0.75,
                     ocr_threshold_multiplier: float = 1.5,
                     use_vlm_fallback: bool = True,
                     confidence_threshold: float = 70.0,
                     verbose: bool = False) -> Tuple[str, str]:
    """
    Extract text from a single PDF page using intelligent hybrid approach.
    
    Strategy:
    1. If text spans most of page height (>vertical_coverage_threshold) → use pdfplumber only
    2. If pdfplumber gets minimal text (<min_text_threshold) → use OCR (image-only page)
    3. Otherwise → run both and compare, using OCR if it captures significantly more content
    4. If OCR used, check quality:
       a) First check OCR confidence score - if too low, use VLM
       b) Then check for complex tables - if detected, use VLM
    
    Args:
        pdf: pdfplumber PDF object
        page_num: Page number (1-indexed)
        pdf_path: Path to PDF file (needed for OCR/VLM)
        min_text_threshold: Minimum chars to consider pdfplumber successful
        vertical_coverage_threshold: Skip OCR if text spans this fraction of page height
        ocr_threshold_multiplier: Use OCR if it produces this much more text
        use_vlm_fallback: Whether to use VLM when OCR quality is poor
        confidence_threshold: Minimum average OCR confidence (0-100) to trust OCR
        
    Returns:
        (text, method) where method is "pdfplumber", "ocr", or "vlm"
    """
    page = pdf.pages[page_num - 1]
    
    # Extract with pdfplumber
    pdfplumber_text = page.extract_text() or ""
    pdfplumber_len = len(pdfplumber_text.strip())
    
    # Case 1: Check if text spans full page height → use pdfplumber (fast check first)
    try:
        words = page.extract_words()
        if words:
            page_height = page.height
            text_top = min(word['top'] for word in words)
            text_bottom = max(word['bottom'] for word in words)
            vertical_span = text_bottom - text_top
            coverage = vertical_span / page_height
            
            if coverage >= vertical_coverage_threshold:
                return pdfplumber_text, "pdfplumber"
    except Exception:
        pass
    
    # Case 2: Minimal extractable text → image-only page, use OCR
    if pdfplumber_len < min_text_threshold:
        ocr_text, ocr_confidence = _run_ocr(pdf_path, page_num)
        if ocr_text:
            return _check_ocr_quality_and_decide(ocr_text, ocr_confidence, pdf_path, page_num,
                                                 use_vlm_fallback, confidence_threshold, verbose)
        return pdfplumber_text, "pdfplumber"
    
    # Case 3: Mixed page → run both and compare
    ocr_text, ocr_confidence = _run_ocr(pdf_path, page_num)
    if ocr_text:
        ocr_len = len(ocr_text.strip())
        
        # Use OCR if it captured significantly more content
        if ocr_len > pdfplumber_len * ocr_threshold_multiplier:
            return _check_ocr_quality_and_decide(ocr_text, ocr_confidence, pdf_path, page_num,
                                                 use_vlm_fallback, confidence_threshold, verbose)
            
    
    return pdfplumber_text, "pdfplumber"


def extract_pdf(pdf_path: str, max_pages: int = None, 
                use_vlm_fallback: bool = True, verbose: bool = False) -> List[Dict]:
    """
    Extract text from all pages of a PDF.
    
    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum pages to process (None = all)
        use_vlm_fallback: Whether to use VLM when OCR quality is poor
        
    Returns:
        List of dicts with 'page', 'text', 'method' for each page
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    results = []
    
    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        if max_pages:
            num_pages = min(num_pages, max_pages)
        
        for i in range(1, num_pages + 1):
            text, method = extract_page_text(pdf, i, pdf_path, 
                                            use_vlm_fallback=use_vlm_fallback,
                                            verbose=verbose)
            results.append({
                'page': i,
                'text': text,
                'method': method
            })
    
    return results


def test_extraction(pdf_path: str, max_pages: int = 2):
    """Test extraction on a PDF and display results."""
    print(f"Testing extraction on: {pdf_path}")
    print("=" * 70)
    
    results = extract_pdf(pdf_path, max_pages=max_pages, verbose=True)
    
    for result in results:
        page = result['page']
        text = result['text']
        method = result['method']
        
        print(f"\nPage {page} [{method}] ({len(text)} chars):")
        print("-" * 70)
        print(text)
    
    # Summary
    methods = [r['method'] for r in results]
    print("\n" + "=" * 70)
    print(f"Summary: {len(results)} pages processed")
    print(f"  pdfplumber: {methods.count('pdfplumber')}")
    print(f"  OCR: {methods.count('ocr')}")
    print(f"  VLM: {methods.count('vlm')}")


if __name__ == "__main__":
    #sample_pdf = "data/raw/CARDIA documentation/Y00/DOC/PROTOCOL/D10758.pdf"
    sample_pdf = "data/raw/CARDIA documentation/Y05/DOC/CAECHO.PDF"
    #sample_pdf = "data/raw/CARDIA documentation/DOC (procedures)/fu144mo.pdf"

    test_extraction(sample_pdf, max_pages=20)


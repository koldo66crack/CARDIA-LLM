"""
Test text extraction from CARDIA PDFs using hybrid approach.
Handles both extractable text and scanned images, with proper column/table support.

Prerequisites:
1. Python packages: pip install pdfplumber pdf2image pytesseract pillow
2. Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki (for scanned pages)
3. Poppler: https://github.com/osber/pdfop/releases (for scanned pages)

Usage:
    python src/test_ocr.py
"""

import os
import pdfplumber

# OCR dependencies (optional, will fallback if not available)
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: OCR libraries not available. Install with: pip install pdf2image pytesseract")

def _run_ocr(pdf_path, page_num):
    """Helper to run OCR on a specific page."""
    if not OCR_AVAILABLE:
        return None
    
    try:
        images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
        if images:
            return pytesseract.image_to_string(images[0])
    except Exception as e:
        print(f"  → OCR failed on page {page_num}: {e}")
    
    return None


def extract_page_text(pdf, page_num, min_text_threshold=100, 
                     vertical_coverage_threshold=0.75, ocr_threshold_multiplier=1.5):
    """
    Extract text from a single PDF page using intelligent hybrid approach.
    
    Strategy:
    1. If pdfplumber gets minimal text (<min_text_threshold) → use OCR (image-only page)
    2. If text spans most of page height (>vertical_coverage_threshold) → use text extraction only
    3. Otherwise → run both and compare, using OCR if it captures significantly more content
    
    Args:
        pdf: pdfplumber PDF object
        page_num: Page number (1-indexed)
        min_text_threshold: Minimum chars to consider pdfplumber successful
        vertical_coverage_threshold: Skip OCR if text spans this fraction of page height
        ocr_threshold_multiplier: Use OCR if it produces this much more text
        
    Returns:
        tuple: (text, method_used) where method is "text_extraction" or "ocr"
    """
    page = pdf.pages[page_num - 1]
    pdf_path = pdf.stream.name if hasattr(pdf.stream, 'name') else None
    
    # Extract with pdfplumber
    pdfplumber_text = page.extract_text() or ""
    pdfplumber_len = len(pdfplumber_text.strip())
    
    # Case 1: Minimal extractable text → image-only page, use OCR
    if pdfplumber_len < min_text_threshold:
        if OCR_AVAILABLE and pdf_path:
            ocr_text = _run_ocr(pdf_path, page_num)
            if ocr_text:
                return ocr_text, "ocr"
        return pdfplumber_text, "text_extraction"
    
    # Case 2: Check if text spans full page height → skip OCR for efficiency
    try:
        words = page.extract_words()
        if words:
            page_height = page.height
            text_top = min(word['top'] for word in words)
            text_bottom = max(word['bottom'] for word in words)
            vertical_span = text_bottom - text_top
            coverage = vertical_span / page_height
            
            if coverage >= vertical_coverage_threshold:
                return pdfplumber_text, "text_extraction"
    except Exception:
        pass
    
    # Case 3: Mixed page → run both and compare
    if OCR_AVAILABLE and pdf_path:
        ocr_text = _run_ocr(pdf_path, page_num)
        if ocr_text:
            ocr_len = len(ocr_text.strip())
            
            # Use OCR if it captured significantly more content
            if ocr_len > pdfplumber_len * ocr_threshold_multiplier:
                return ocr_text, "ocr"
    
    return pdfplumber_text, "text_extraction"


def test_pdf_extraction(pdf_path: str, max_pages: int = 2):
    """
    Test text extraction on a PDF using hybrid pdfplumber + OCR approach.
    
    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to process
        
    Returns:
        list: List of (text, method) tuples for each page
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    print(f"Testing extraction on: {pdf_path}")
    
    results = []
    with pdfplumber.open(pdf_path) as pdf:
        num_pages = min(len(pdf.pages), max_pages)
        print(f"Processing {num_pages} page(s)...\n")
        
        for i in range(1, num_pages + 1):
            print(f"Page {i}...", end=" ")
            text, method = extract_page_text(pdf, i)
            results.append((text, method))
            print(f"({len(text)} characters via {method})")
    
    return results


def main():
    sample_pdf = "data/raw/CARDIA documentation/Y00/DOC/Aaf10.PDF"
    #sample_pdf = "data/raw/CARDIA documentation/Y00/DOC/PROTOCOL/D10766.PDF"
    
    print("=" * 70)
    print("CARDIA PDF EXTRACTION TEST (pdfplumber + OCR hybrid)")
    print("=" * 70 + "\n")
    
    try:
        results = test_pdf_extraction(sample_pdf, max_pages=20)
        
        # Calculate statistics
        total_chars = sum(len(text) for text, _ in results)
        methods_used = [method for _, method in results]
        text_extraction_count = methods_used.count("text_extraction")
        ocr_count = methods_used.count("ocr")
        
        print("\n" + "=" * 70)
        print("EXTRACTION SUCCESSFUL!")
        print("=" * 70)
        print(f"Total characters extracted: {total_chars:,}")
        print(f"Methods used: {text_extraction_count} text_extraction, {ocr_count} ocr")
        print("=" * 70 + "\n")
        
        # Show preview of all pages
        if results:
            for page_num, (text, method) in enumerate(results, 1):
                print(f"Page {page_num} [{method}]:")
                print(text)
                print("-" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"EXTRACTION FAILED: {e}")
        print("=" * 70)


if __name__ == "__main__":
    main()


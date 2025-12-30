"""
CARDIA Data Preprocessing Module

This module handles all data preparation:
- CSV data dictionary processing (BIOLINCC main + ancillary studies)
- PDF documentation extraction (OCR + pdfplumber + VLM fallback)
- Word document extraction (python-docx + Word COM)
"""

from preprocessing.csv_preprocessor import preprocess_biolincc_csv, save_combined_jsonl
from preprocessing.pdf_extractor import extract_page_text, extract_pdf
from preprocessing.doc_extractor import extract_word_document_chunked
from preprocessing.ocr_quality import should_use_vlm
from preprocessing.vlm_utils import extract_with_vlm
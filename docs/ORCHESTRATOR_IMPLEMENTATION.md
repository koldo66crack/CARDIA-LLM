# CARDIA Documentation Orchestrator Implementation Guide

## Overview

This document outlines the implementation of the orchestrator script that will process all CARDIA documentation files (PDFs and Word documents) and create a structured JSONL database for RAG integration.

## Current State

### Completed Components

1. **PDF Text Extraction** (`preprocessing/pdf_extractor.py`)
   - 3-tier extraction approach:
     - **Tier 1**: `pdfplumber` for text-based PDFs with layout preservation
     - **Tier 2**: Tesseract OCR for image-based content
     - **Tier 3**: Gemini Vision (VLM) for complex tables/diagrams (triggered by quality heuristics)
   - Decision logic:
     - Use OCR if pdfplumber output < 100 characters
     - Skip OCR if text spans > 75% of page height
     - Prefer OCR if output length ≥ 1.5x pdfplumber's output
   - Quality checks before VLM:
     - Short lines: > 15% of lines have ≤ 2 words (excluding lines ending in `.`, `,`, `;`)
     - Table indicators: ≥ 5 instances of "Num" or "Char"

2. **Word Document Extraction** (`preprocessing/doc_extractor.py`)
   - `.docx`: Uses `python-docx` (native Python library)
   - `.doc`: Uses Microsoft Word COM automation (`pywin32`)
     - Converts `.doc` → `.docx` temporarily
     - Extracts with `python-docx`
   - Handles both paragraphs and tables

3. **OCR Quality Heuristics** (`preprocessing/ocr_quality.py`)
   - `has_too_many_short_lines()`: Detects fragmented text from tables
   - `has_table_indicators()`: Detects table-specific keywords
   - `should_use_vlm()`: Combined decision function

4. **VLM Utilities** (`preprocessing/vlm_utils.py`)
   - Gemini Vision integration for complex page extraction
   - Custom prompt for preserving structure (tables, diagrams, questionnaires)

## Source Data Structure

```
data/raw/CARDIA documentation/
├── Y00/         # Wave 0 (baseline)
├── Y01/         # Wave 1 (year 2)
├── Y05/         # Wave 2 (year 5)
├── Y07/         # Wave 3 (year 7)
├── Y10/         # Wave 4 (year 10)
├── Y15/         # Wave 5 (year 15)
├── Y20/         # Wave 6 (year 20)
├── Y25/         # Wave 7 (year 25)
└── Y30/         # Wave 8 (year 30)
```

Each wave folder contains:
- `DOC/PROTOCOL/`: General study protocols (broad context)
- `DOC/MOO/`: Manuals of Operations (procedural details)
- `DOC/*.pdf` or `DOC/*.doc`: Dataset-specific documentation (~1-to-1 with dataset names)

### File Naming Conventions

- **Dataset-related PDFs**: Sometimes match dataset names (e.g., `aachem.pdf` for dataset `aachem`), but often use arbitrary IDs (e.g., `D10758.pdf`)
- **Wave naming**: `Y00` through `Y30` (0, 2, 5, 7, 10, 15, 20, 25, 30 years)
- **Dataset naming**: `{wave}{version}{content}` (e.g., `aachem` = wave A, version A, chemistry)

### File Types

- PDFs: Mix of text-based and image-based (scanned) content
- Word: Both `.doc` (old binary format) and `.docx` files
- Average dataset PDF: ~3 pages
- Protocol PDFs: Up to 90 pages

## Target Output Format

### File Location
`data/processed/cardia_pdf_documentation.jsonl`

### JSONL Structure

Each line represents **one page** from a document:

```json
{
  "id": "Y00_D10758_1",
  "source_file": "data/raw/CARDIA documentation/Y00/DOC/PROTOCOL/D10758.pdf",
  "page": 1,
  "wave": "Y00",
  "doc_type": "PROTOCOL",
  "method": "pdfplumber",
  "content": "extracted text content...",
  "year": 0,
  "filename": "D10758"
}
```

### Field Definitions

- **id**: `{wave}_{filename}_{page}` (unique identifier)
- **source_file**: Full relative path to original document
- **page**: Page number (1-indexed)
- **wave**: Wave folder name (Y00, Y01, Y05, etc.)
- **doc_type**: Document category
  - `"PROTOCOL"`: General study protocols
  - `"MOO"`: Manuals of Operations
  - `"DATASET"`: Dataset-specific documentation
  - `"OTHER"`: Miscellaneous documents
- **method**: Extraction method used
  - `"pdfplumber"`: Text-based extraction
  - `"ocr"`: Tesseract OCR
  - `"vlm"`: Gemini Vision
  - `"docx"`: python-docx extraction
  - `"word_com"`: Word COM automation
- **content**: Extracted text from the page
- **year**: Numeric year (0, 2, 5, 7, 10, 15, 20, 25, 30) mapped from wave
- **filename**: Document filename without extension

### Wave-to-Year Mapping

```python
WAVE_TO_YEAR = {
    "Y00": 0,
    "Y01": 2,
    "Y05": 5,
    "Y07": 7,
    "Y10": 10,
    "Y15": 15,
    "Y20": 20,
    "Y25": 25,
    "Y30": 30
}
```

### Document Type Detection

Infer from path structure:
- Path contains `/PROTOCOL/` → `"PROTOCOL"`
- Path contains `/MOO/` → `"MOO"`
- File is in root `DOC/` directory → `"DATASET"`
- Otherwise → `"OTHER"`

## Orchestrator Script Requirements

### Script Name
`preprocessing/process_all_docs.py`

### Core Functionality

1. **Directory Traversal**
   - Walk through all wave folders (Y00-Y30)
   - Find all PDF and Word documents recursively
   - Track progress with simple status messages

2. **File Processing**
   - For each PDF: Call `pdf_extractor.extract_pdf()`
   - For each Word doc: Call `doc_extractor.extract_word_document()`
   - Handle errors gracefully (log failures, continue processing)

3. **Metadata Extraction**
   - Extract wave from folder name
   - Map wave to year
   - Infer doc_type from path
   - Extract filename (without extension)

4. **JSONL Generation**
   - Create one chunk per page
   - Generate unique IDs
   - Write incrementally (don't hold all in memory)

5. **Progress Tracking**
   - Print: `"Processing: {filename} ({current}/{total})"` for each file
   - Print summary at end: `"Processed {total} files, {pages} pages"`
   - Log any failures to console

### Error Handling

- **Skip failed files**: Continue processing other files if one fails
- **Print errors**: Show which file failed and why
- **No fallbacks**: If extraction libraries fail to import, script should crash immediately
- **Validation**: Ensure content is not empty before writing to JSONL

### Implementation Notes

1. **No conditional imports**: All dependencies must be available
2. **Keep it simple**: No complex abstractions or unnecessary logging
3. **Memory efficient**: Write to JSONL incrementally, not all at once
4. **Deterministic ordering**: Process files in sorted order for reproducibility

### Example Processing Flow

```
Processing: Y00/DOC/PROTOCOL/D10758.pdf (1/500)
  → 12 pages extracted (pdfplumber: 8, ocr: 2, vlm: 2)
Processing: Y00/DOC/aachem.pdf (2/500)
  → 3 pages extracted (pdfplumber: 3)
Processing: Y20/DOC/GAF22.DOC (3/500)
  → 1 page extracted (word_com: 1)
...
```

## Integration with RAG Pipeline

After the orchestrator script completes:

1. **Indexing** (`src/index.py`)
   - Load `cardia_pdf_documentation.jsonl`
   - Embed each chunk using BGE model
   - Build/extend FAISS index
   - Combine with existing `biolincc_data_dictionary.jsonl` index

2. **Retrieval** (`src/rag_retriever.py`)
   - Implement keyword search for dataset/variable names
   - Return entire PDF if keyword match found (not just matching chunk)
   - Combine semantic search (FAISS) with keyword matching

3. **Sufficiency Check** (new component)
   - After initial retrieval, use Gemini Flash to check if info is sufficient
   - Output: `{"sufficient": bool, "search_query": str}`
   - If insufficient, trigger PDF search with generated query

4. **Response Generation** (`src/llm.py`)
   - Distinguish between variable data and PDF documentation in responses
   - Cite source PDFs with clickable links
   - Show "Sources Used" section at end

## Dependencies

Ensure these are in `requirements.txt`:

```
pdfplumber>=0.9.0
pdf2image>=1.16.0
pytesseract>=0.3.10
pillow>=9.0.0
python-docx>=0.8.11
pywin32>=306  # For Word COM automation
google-generativeai  # For Gemini Vision
python-dotenv
```

System dependencies:
- Tesseract OCR (must be in PATH)
- Microsoft Word (for `.doc` files on Windows)

## Testing

Before running the full orchestrator:

1. Test on a single wave folder first
2. Verify JSONL output structure
3. Check that all `doc_type` values are correctly assigned
4. Ensure page numbering is correct
5. Validate that extraction methods are logged properly

## Open Questions for New Chat

1. Should the orchestrator be idempotent (skip already-processed files)?
2. How should we handle duplicate filenames across waves?
3. Should we validate that dataset names in filenames match actual dataset names in BIOLINCC?
4. Do we need a separate index for PDF documentation vs. variable dictionary, or combine them?


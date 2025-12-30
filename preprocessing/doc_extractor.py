"""
DOC/DOCX Text Extraction for CARDIA Documentation.

Extracts Word documents as chunks for RAG indexing:
- .docx: Uses python-docx (native Python library)
- .doc: Uses Microsoft Word COM automation (pywin32) on Windows
  Converts .doc → .docx temporarily, then extracts with python-docx

Paragraph-based chunking:
- Paragraphs grouped to target 800-1200 chars per chunk
- Tables kept as single chunks (never split)
- Maintains document order via chunk indices
"""

import os
import tempfile
import shutil
import time
from pathlib import Path
from typing import List, Dict
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl


def _extract_docx_elements(file_path: str) -> List[Dict]:
    """
    Extract paragraphs and tables from .docx as structured elements.
    
    Preserves original document order by iterating through body elements.
    Returns list of dicts with 'type' ('paragraph' or 'table') and 'text'.
    Tables are kept as single elements to avoid splitting them across chunks.
    """
    doc = Document(file_path)
    elements = []
    
    # Iterate through body elements in original order
    for element in doc.element.body:
        if isinstance(element, CT_P):
            # It's a paragraph
            para = Paragraph(element, doc)
            text = para.text.strip()
            if text:
                elements.append({'type': 'paragraph', 'text': text})
                
        elif isinstance(element, CT_Tbl):
            # It's a table
            table = Table(element, doc)
            rows = []
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    rows.append(row_text)
            if rows:
                elements.append({'type': 'table', 'text': '\n'.join(rows)})
    
    return elements


def _chunk_elements(elements: List[Dict], min_chars: int = 800, max_chars: int = 1200) -> List[str]:
    """
    Group elements into chunks of roughly min_chars to max_chars.
    
    Rules:
    - Never split paragraphs or tables mid-text
    - Tables always get their own chunk (may exceed max_chars)
    - Paragraphs are grouped until we hit the target range
    """
    chunks = []
    current_chunk = []
    current_len = 0
    
    for elem in elements:
        text = elem['text']
        text_len = len(text)
        
        # Tables get their own chunk
        if elem['type'] == 'table':
            # Save current chunk if not empty
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_len = 0
            # Add table as its own chunk
            chunks.append(text)
            continue
        
        # For paragraphs, group until we hit target size
        if current_len + text_len > max_chars and current_chunk:
            # Current chunk is full enough, save it
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [text]
            current_len = text_len
        else:
            current_chunk.append(text)
            current_len += text_len
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks
        

def _convert_doc_to_docx(file_path: str, max_retries: int = 3) -> str:
    """
    Convert .doc to .docx using Word COM and return path to temp .docx.
    
    Note: Returns path within a temp directory that caller must manage.
    
    Args:
        file_path: Path to .doc file
        max_retries: Number of retry attempts for COM automation (default: 3)
    """
    try:
        import win32com.client
        import pythoncom
    except ImportError:
        raise RuntimeError(
            "pywin32 not installed. Install it with:\n"
            "  pip install pywin32"
        )
    
    abs_path = os.path.abspath(file_path)
    temp_dir = tempfile.mkdtemp()
    temp_docx = os.path.join(temp_dir, "temp_converted.docx")
    
    last_error = None
    
    for attempt in range(max_retries):
        word = None
        doc = None
        try:
            # Initialize COM for this thread (safe to call multiple times)
            pythoncom.CoInitialize()
            
            # Create new Word instance with DispatchEx (creates fresh instance)
            word = win32com.client.DispatchEx("Word.Application")
            word.Visible = False
            word.DisplayAlerts = False
            
            # Small delay to let Word initialize
            time.sleep(0.5)
            
            doc = word.Documents.Open(abs_path, ReadOnly=True)
            doc.SaveAs2(os.path.abspath(temp_docx), FileFormat=12)
            doc.Close(SaveChanges=False)
            
            # Success - clean up and return
            word.Quit()
            pythoncom.CoUninitialize()
            return temp_docx
            
        except Exception as e:
            last_error = e
            # Clean up on failure
            if doc:
                try:
                    doc.Close(SaveChanges=False)
                except:
                    pass
            if word:
                try:
                    word.Quit()
                except:
                    pass
            try:
                pythoncom.CoUninitialize()
            except:
                pass
            
            # If not last attempt, wait before retry
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
    
    # All retries failed
    raise RuntimeError(f"Word COM automation for file {file_path} failed after {max_retries} attempts: {str(last_error)}")


def _extract_doc_elements_with_word_com(file_path: str) -> List[Dict]:
    """Extract structured elements from .doc files via Word COM."""
    temp_docx = _convert_doc_to_docx(file_path)
    try:
        return _extract_docx_elements(temp_docx)
    finally:
        shutil.rmtree(os.path.dirname(temp_docx), ignore_errors=True)


def extract_word_document_chunked(file_path: str, 
                                   min_chars: int = 800, 
                                   max_chars: int = 1200) -> List[Dict]:
    """
    Extract text from a Word document as chunks for RAG indexing.
    
    Chunks are created using paragraph-based grouping:
    - Paragraphs grouped to target 800-1200 chars per chunk
    - Tables kept as single chunks (never split)
    - Maintains document order via 'page' field (1-indexed)
    
    Args:
        file_path: Path to Word document
        min_chars: Minimum chars per chunk (soft limit)
        max_chars: Maximum chars per chunk (soft limit, tables may exceed)
        
    Returns:
        List of dicts with 'page', 'text', 'method' for each chunk
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    suffix = Path(file_path).suffix.lower()
    
    # Extract structured elements
    if suffix == '.docx':
        elements = _extract_docx_elements(file_path)
    elif suffix == '.doc':
        elements = _extract_doc_elements_with_word_com(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Expected .doc or .docx")
    
    # Chunk the elements
    chunks = _chunk_elements(elements, min_chars, max_chars)
    
    # Format as page-like records
    results = []
    for i, chunk_text in enumerate(chunks, start=1):
        results.append({
            'page': i,
            'text': chunk_text,
            'method': 'word'
        })
    
    return results


def test_extraction(file_path: str):
    """Test extraction on a Word document and display results."""
    print(f"Testing extraction on: {file_path}")
    print("=" * 70)
    
    try:
        # Test chunked extraction
        chunks = extract_word_document_chunked(file_path)
        
        total_chars = sum(len(c['text']) for c in chunks)
        print(f"\nExtracted {len(chunks)} chunks, {total_chars} total characters")
        print("=" * 70)
        
        for chunk in chunks:
            print(f"\n--- Chunk {chunk['page']} ({len(chunk['text'])} chars) ---")
            print(chunk['text'])
        
    except Exception as e:
        print(f"\n[ERROR] Extraction failed: {str(e)}")
        return False
    
    return True


if __name__ == "__main__":
    #sample_doc = "data/raw/CARDIA documentation/Y25/DOC/HAF08.DOC"
    sample_doc = "data/raw/CARDIA documentation/DOC (procedures)/FADEATH.doc"
    test_extraction(sample_doc)


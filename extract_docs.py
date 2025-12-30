"""
CARDIA Documentation Extraction Orchestrator

Extracts text from all PDFs and Word documents in the CARDIA documentation folder.
Outputs a JSONL file for RAG indexing.

Usage:
    python extract_docs.py          # Process all documents
    python extract_docs.py --test   # Process 6 test files from Y00
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Lock

from preprocessing.pdf_extractor import extract_pdf
from preprocessing.doc_extractor import extract_word_document_chunked

# File lock for JSONL writes
_write_lock = Lock()

# Configuration
SOURCE_DIR = "data/raw/CARDIA documentation"
OUTPUT_FILE = "data/processed/cardia_documentation.jsonl"

WAVE_TO_YEAR = {
    "Y00": 0, "Y01": 2, "Y05": 5, "Y07": 7, "Y10": 10,
    "Y15": 15, "Y20": 20, "Y25": 25, "Y30": 30
}

SUPPORTED_EXTENSIONS = {'.pdf', '.doc', '.docx'}


def get_doc_type(file_path: str) -> str:
    """Infer document type from path."""
    path_lower = file_path.lower()
    if '/protocol/' in path_lower or '\\protocol\\' in path_lower:
        return "PROTOCOL"
    elif '/moo/' in path_lower or '\\moo\\' in path_lower:
        return "MOO"
    elif '/doc/' in path_lower or '\\doc\\' in path_lower:
        return "DATASET"
    return "OTHER"


def get_wave_from_path(file_path: str) -> str:
    """Extract wave (Y00, Y01, etc.) from path."""
    for wave in WAVE_TO_YEAR.keys():
        if f"/{wave}/" in file_path or f"\\{wave}\\" in file_path:
            return wave
    return "UNKNOWN"


def load_processed_files(output_file: str) -> Set[str]:
    """Load set of already-processed source files from existing JSONL."""
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    processed.add(record.get('source_file', ''))
                except json.JSONDecodeError:
                    continue
    return processed


def discover_files(source_dir: str) -> List[str]:
    """
    Discover all PDF and Word documents in source directory.
    
    Args:
        source_dir: Root directory to search
    """
    files = []
    
    for root, _, filenames in os.walk(source_dir):
        for filename in sorted(filenames):
            ext = Path(filename).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            
            file_path = os.path.join(root, filename)
            files.append(file_path)
    
    return sorted(files)


def get_test_files(source_dir: str) -> List[str]:
    """
    Get 6 test files from Y00: 2 from DOC root, 2 from MOO, 2 from PROTOCOL.
    """
    test_files = []
    y00_doc = os.path.join(source_dir, "Y00", "DOC")
    
    # 2 from DOC root (dataset-specific)
    doc_root_files = [
        f for f in os.listdir(y00_doc)
        if os.path.isfile(os.path.join(y00_doc, f)) and 
           Path(f).suffix.lower() in SUPPORTED_EXTENSIONS
    ][:2]
    test_files.extend([os.path.join(y00_doc, f) for f in doc_root_files])
    
    # 2 from MOO
    moo_dir = os.path.join(y00_doc, "MOO")
    if os.path.exists(moo_dir):
        moo_files = [
            f for f in os.listdir(moo_dir)
            if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS
        ][:2]
        test_files.extend([os.path.join(moo_dir, f) for f in moo_files])
    
    # 2 from PROTOCOL
    protocol_dir = os.path.join(y00_doc, "PROTOCOL")
    if os.path.exists(protocol_dir):
        protocol_files = [
            f for f in os.listdir(protocol_dir)
            if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS
        ][:2]
        test_files.extend([os.path.join(protocol_dir, f) for f in protocol_files])
    
    return test_files


def process_file(file_path: str, verbose: bool = False) -> List[Dict]:
    """
    Process a single file (PDF or Word) and return list of page/chunk records.
    """
    ext = Path(file_path).suffix.lower()
    
    if ext == '.pdf':
        return extract_pdf(file_path, verbose=verbose)
    elif ext in {'.doc', '.docx'}:
        return extract_word_document_chunked(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def process_single_file(file_path: str, verbose: bool = False) -> Tuple[Optional[List[Dict]], Optional[str], Dict]:
    """
    Process a single file and return its records.
    
    Returns:
        (records, error, stats): 
            - records: List of page records if successful, None if error
            - error: Error message if failed, None if successful
            - stats: Dict with 'pages' count and method counts
    """
    try:
        # Extract pages
        pages = process_file(file_path, verbose=verbose)
        
        # Get metadata
        wave = get_wave_from_path(file_path)
        doc_type = get_doc_type(file_path)
        filename = Path(file_path).name
        
        # Create records
        records = []
        for page_data in pages:
            record = create_record(file_path, page_data, wave, doc_type, filename)
            records.append(record)
        
        # Compute stats
        methods = {}
        for p in pages:
            m = p['method']
            methods[m] = methods.get(m, 0) + 1
        
        stats = {
            'pages': len(pages),
            'methods': methods
        }
        
        return records, None, stats
        
    except Exception as e:
        return None, str(e), {}


def create_record(file_path: str, page_data: Dict, wave: str, 
                  doc_type: str, filename: str) -> Dict:
    """Create a JSONL record for a single page/chunk."""
    page_num = page_data['page']
    return {
        "id": f"{wave}_{filename}_{page_num}",
        "source_file": file_path.replace("\\", "/"),
        "wave": wave,
        "year": WAVE_TO_YEAR.get(wave, -1),
        "filename": filename,
        "doc_type": doc_type,
        "page": page_num,
        "method": page_data['method'],
        "content": page_data['text'],
    }


def main():
    parser = argparse.ArgumentParser(description="Extract text from CARDIA documentation")
    parser.add_argument('--test', action='store_true', 
                        help="Process only 6 test files from Y00")
    parser.add_argument('--workers', type=int, default=4,
                        help="Number of parallel workers (default: 4, set to 1 to disable)")
    parser.add_argument('--verbose', action='store_true',
                        help="Print detailed extraction messages")
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("CARDIA DOCUMENTATION EXTRACTION")
    print("=" * 70)
    
    # Discover files
    if args.test:
        print("\n[TEST MODE] Processing 6 files from Y00...")
        files = get_test_files(SOURCE_DIR)
    else:
        print(f"\nDiscovering files in {SOURCE_DIR}...")
        files = discover_files(SOURCE_DIR)
    
    print(f"Found {len(files)} files to process")
    
    # Load already-processed files for idempotency
    processed_files = load_processed_files(OUTPUT_FILE)
    if processed_files:
        print(f"Skipping {len(processed_files)} already-processed files")
    
    # Filter out already-processed files
    files_to_process = [
        f for f in files 
        if f.replace("\\", "/") not in processed_files
    ]
    
    if not files_to_process:
        print("\nNo new files to process. Exiting.")
        return 0
    
    print(f"\nProcessing {len(files_to_process)} new files with {args.workers} worker(s)...")
    print("-" * 70)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Process files in parallel
    total_pages = 0
    processed_count = 0
    executor = None
    
    try:
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_file:
            # Submit all jobs
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                # Map futures to file paths for tracking
                future_to_file = {
                    executor.submit(process_single_file, file_path, args.verbose): file_path
                    for file_path in files_to_process if file_path!="data/raw/CARDIA documentation\Y01\DOC\MOO\D10687.PDF"
                }
                
                # Process results as they complete
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    filename = Path(file_path).stem
                    doc_type = get_doc_type(file_path)
                    
                    processed_count += 1
                    print(f"[{processed_count}/{len(files_to_process)}] {filename} ({doc_type})...", end=" ", flush=True)
                    
                    try:
                        records, error, stats = future.result()
                        
                        if error:
                            print(f"FAILED - {error}")
                            print(f"\n[ERROR] Stopping due to failure on {file_path}")
                            print(f"Progress saved: {processed_count - 1} files, {total_pages} pages")
                            print("Rerun the script to continue from where you left off.")
                            executor.shutdown(wait=False, cancel_futures=True)
                            return 1
                        
                        # Write records atomically with lock
                        with _write_lock:
                            for record in records:
                                out_file.write(json.dumps(record) + "\n")
                            # Flush after each file for safety
                            out_file.flush()
                        
                        # Display stats
                        methods = stats.get('methods', {})
                        method_str = ", ".join(f"{m}: {c}" for m, c in methods.items())
                        print(f"{stats['pages']} pages ({method_str})")
                        total_pages += stats['pages']
                        
                    except Exception as e:
                        print(f"FAILED - {str(e)}")
                        print(f"\n[ERROR] Stopping due to failure on {file_path}")
                        print(f"Progress saved: {processed_count - 1} files, {total_pages} pages")
                        print("Rerun the script to continue from where you left off.")
                        executor.shutdown(wait=False, cancel_futures=True)
                        return 1
        
        # Summary
        print("\n" + "=" * 70)
        print("EXTRACTION COMPLETE")
        print("=" * 70)
        print(f"Processed: {processed_count} files, {total_pages} pages")
        print(f"Output: {OUTPUT_FILE}")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n\n[INTERRUPTED] Shutting down workers...")
        if executor:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except:
                pass
        print(f"Progress saved: {processed_count} files, {total_pages} pages")
        print("Rerun the script to continue from where you left off.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


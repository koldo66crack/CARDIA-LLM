"""
CARDIA RAG Pipeline Setup Orchestrator
Single entry point for preprocessing and indexing both BIOLINCC and Ancillary data.
Run once to set up the vector database for retrieval.

Usage:
    python setup.py
"""

import os
import sys
import json
from src.preprocess import preprocess_biolincc_csv, save_combined_jsonl
from src.index import build_index_from_jsonl

def main():
    """
    Orchestrate the complete setup pipeline:
    1. Preprocess both BIOLINCC main and ancillary studies into JSONL
    2. Build FAISS vector index from combined JSONL
    """
    print("\n" + "=" * 70)
    print("CARDIA RAG PIPELINE SETUP")
    print("=" * 70 + "\n")
    
    try:
        # Step 1: Preprocessing
        all_chunks = []
        all_dataframes = []
        
        # Process main study
        main_study_csv = "data/raw/BIOLINCC_Main Study Data Dictionary.csv"
        if not os.path.exists(main_study_csv):
            print(f"[ERROR] Main study CSV not found: {main_study_csv}")
            return 1
        
        main_chunks, main_df = preprocess_biolincc_csv(main_study_csv, study_type="main")
        all_chunks.extend(main_chunks)
        all_dataframes.append(main_df)
        
        # Process ancillary studies
        ancillary_csv = "data/raw/Ancillary Studies Data Dictionary - cleaned.csv"
        if not os.path.exists(ancillary_csv):
            print(f"[ERROR] Ancillary studies CSV not found: {ancillary_csv}")
            return 1
        
        ancillary_chunks, ancillary_df = preprocess_biolincc_csv(ancillary_csv, study_type="ancillary")
        all_chunks.extend(ancillary_chunks)
        all_dataframes.append(ancillary_df)
        
        # Save combined JSONL
        jsonl_path = save_combined_jsonl(all_chunks, all_dataframes)
        
        # Step 2: Indexing
        if not os.path.exists(jsonl_path):
            print(f"[ERROR] JSONL file not found: {jsonl_path}")
            return 1
        
        build_index_from_jsonl(jsonl_path)
        
        # Step 3: Load summary statistics
        summary_path = os.path.join("data/processed", "preprocessing_summary.json")
        if not os.path.exists(summary_path):
            print(f"[ERROR] Summary file not found: {summary_path}")
            return 1
        
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        # Display summary
        print("\n" + "=" * 70)
        print("SETUP COMPLETE")
        print("=" * 70)
        print(f"\nTotal chunks indexed: {summary['total_chunks']}")
        print(f"  - Main study: {summary['main_study_chunks']} chunks")
        print(f"  - Ancillary studies: {summary['ancillary_study_chunks']} chunks")
        print(f"\nVector index ready for retrieval!")
        print(f"Run: python app.py\n")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

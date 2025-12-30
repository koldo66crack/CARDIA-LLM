"""
Build embeddings and vector index from processed JSONL chunks.
Creates searchable vector database for retrieval using FAISS and BGE embeddings.
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from tqdm import tqdm

# Default paths
VARIABLES_JSONL = "data/processed/biolincc_data_dictionary.jsonl"
DOCS_JSONL = "data/processed/cardia_documentation.jsonl"
OUTPUT_DIR = "data/processed"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def load_jsonl_data(jsonl_path):
    """
    Load and parse JSONL data from processed file.
    
    Args:
        jsonl_path: Path to the JSONL file
        
    Returns:
        List of chunk dictionaries
    """
    print(f"Loading JSONL data from: {jsonl_path}")
    
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                chunk = json.loads(line.strip())
                chunks.append(chunk)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON on line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(chunks)} chunks")
    return chunks


def initialize_embedding_model(model_name=DEFAULT_MODEL):
    """
    Initialize the embedding model.
    
    Args:
        model_name: Name of the sentence transformer model
        
    Returns:
        Initialized SentenceTransformer model
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def extract_content_for_embedding(chunks, enrich_docs=False):
    """
    Extract content fields from chunks for embedding.
    
    Args:
        chunks: List of chunk dictionaries
        enrich_docs: If True, prepend metadata to doc content for better searchability
        
    Returns:
        List of content strings
    """
    print("Extracting content fields for embedding...")
    
    content_strings = []
    for chunk in chunks:
        content = chunk.get('content', '')
        
        if enrich_docs:
            # Prepend metadata for documentation chunks
            content = _enrich_doc_content(chunk)
        
        if content:
            content_strings.append(content)
        else:
            print(f"Warning: Chunk {chunk.get('id', 'unknown')} has no content field")
            content_strings.append("")  # Empty string for missing content
    
    print(f"Extracted {len(content_strings)} content strings")
    return content_strings


def _enrich_doc_content(chunk):
    """
    Enrich documentation chunk content with metadata for better semantic search.
    Prepends filename, wave, year, and doc_type to the content.
    
    Args:
        chunk: Documentation chunk dictionary
        
    Returns:
        Enriched content string
    """
    parts = []
    
    filename = chunk.get('filename', '')
    if filename:
        parts.append(f"Document: {filename}")
    
    wave = chunk.get('wave', '')
    year = chunk.get('year', -1)
    if wave and year != -1:
        parts.append(f"Wave: {wave} (Year {year})")
    elif wave:
        parts.append(f"Wave: {wave}")
    
    doc_type = chunk.get('doc_type', '')
    if doc_type:
        parts.append(f"Type: {doc_type}")
    
    page = chunk.get('page', '')
    if page:
        parts.append(f"Page: {page}")
    
    # Add the actual content
    content = chunk.get('content', '')
    if content:
        parts.append(f"Content: {content}")
    
    return " | ".join(parts)


def generate_embeddings(model, content_strings, batch_size=32):
    """
    Generate embeddings for all content strings.
    
    Args:
        model: SentenceTransformer embedding model
        content_strings: List of content strings
        batch_size: Batch size for processing
        
    Returns:
        numpy array of embeddings
    """
    print(f"Generating embeddings for {len(content_strings)} content strings...")
    print(f"Using batch size: {batch_size}\n")
    
    # Process in batches with tqdm progress bar
    all_embeddings = []
    batch_indices = range(0, len(content_strings), batch_size)
    
    for i in tqdm(batch_indices, desc="Embedding batches", unit="batch"):
        batch = content_strings[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_numpy=True)
        all_embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(all_embeddings)
    print(f"\nGenerated embeddings with shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings):
    """
    Build FAISS index with embeddings.
    
    Args:
        embeddings: numpy array of embeddings
        
    Returns:
        FAISS index
    """
    print("Building FAISS index...")
    
    dimension = embeddings.shape[1]
    print(f"Embedding dimension: {dimension}")
    
    # Create FAISS index (using IndexFlatIP for cosine similarity)
    index = faiss.IndexFlatIP(dimension)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Add embeddings to index
    index.add(embeddings)
    
    print(f"FAISS index built with {index.ntotal} vectors")
    return index


def save_index_and_metadata(index, metadata, output_dir=OUTPUT_DIR, prefix="", model_name=DEFAULT_MODEL):
    """
    Save FAISS index and metadata to disk.
    
    Args:
        index: FAISS index
        metadata: List of metadata chunks
        output_dir: Output directory
        prefix: Prefix for output files (e.g., "docs_" for documentation index)
        model_name: Name of the embedding model used
        
    Returns:
        Tuple of (index_path, metadata_path, info_path)
    """
    print(f"Saving index and metadata to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Build file paths with optional prefix
    index_path = os.path.join(output_dir, f"{prefix}faiss_index.bin")
    metadata_path = os.path.join(output_dir, f"{prefix}metadata.pkl")
    info_path = os.path.join(output_dir, f"{prefix}index_info.json")
    
    # Save FAISS index
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to: {index_path}")
    
    # Save metadata
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved to: {metadata_path}")
    
    # Save index info
    info = {
        "total_vectors": index.ntotal,
        "embedding_dimension": index.d,
        "index_type": "IndexFlatIP",
        "similarity_metric": "cosine",
        "model_name": model_name
    }
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    print(f"Index info saved to: {info_path}")
    
    return index_path, metadata_path, info_path


def build_index(jsonl_path, output_dir=OUTPUT_DIR, prefix="", model_name=DEFAULT_MODEL):
    """
    Build index from a JSONL file.
    
    Args:
        jsonl_path: Path to JSONL file
        output_dir: Output directory for index files
        prefix: Prefix for output files (e.g., "docs_")
        model_name: Name of embedding model
        
    Returns:
        Tuple of (index, metadata, model)
    """
    is_docs = prefix == "docs_"
    index_type = "DOCUMENTATION" if is_docs else "VARIABLES"
    
    print("=" * 60)
    print(f"BUILDING {index_type} INDEX")
    print("=" * 60)
    
    # Step 1: Load JSONL data
    print("\n[STEP 1/5] Loading JSONL data...")
    chunks = load_jsonl_data(jsonl_path)
    
    # Step 2: Initialize embedding model
    print("\n[STEP 2/5] Initializing embedding model...")
    model = initialize_embedding_model(model_name)
    
    # Step 3: Extract content for embedding (enrich docs with metadata)
    print("\n[STEP 3/5] Extracting content fields...")
    content_strings = extract_content_for_embedding(chunks, enrich_docs=is_docs)
    
    # Step 4: Generate embeddings
    print("\n[STEP 4/5] Generating embeddings...")
    embeddings = generate_embeddings(model, content_strings)
    
    # Step 5: Build and save FAISS index
    print("\n[STEP 5/5] Building and saving FAISS index...")
    index = build_faiss_index(embeddings)
    index_path, metadata_path, info_path = save_index_and_metadata(
        index, chunks, output_dir, prefix, model_name
    )
    
    print("\n" + "=" * 60)
    print(f"{index_type} INDEX COMPLETE")
    print("=" * 60)
    print(f"Index file: {index_path}")
    print(f"Metadata file: {metadata_path}")
    print(f"Info file: {info_path}")
    
    return index, chunks, model


def build_variables_index():
    """Build index for BIOLINCC variables."""
    if not os.path.exists(VARIABLES_JSONL):
        print(f"Variables JSONL not found: {VARIABLES_JSONL}")
        return None
    return build_index(VARIABLES_JSONL, OUTPUT_DIR, prefix="")


def build_docs_index():
    """Build index for CARDIA documentation."""
    if not os.path.exists(DOCS_JSONL):
        print(f"Documentation JSONL not found: {DOCS_JSONL}")
        return None
    return build_index(DOCS_JSONL, OUTPUT_DIR, prefix="docs_")


# Keep for backward compatibility
def build_index_from_jsonl(jsonl_path, output_dir=OUTPUT_DIR, model_name=DEFAULT_MODEL):
    """Backward compatible wrapper for build_index."""
    return build_index(jsonl_path, output_dir, prefix="", model_name=model_name)


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        target = sys.argv[1].lower()
        if target == "docs":
            build_docs_index()
        elif target == "variables":
            build_variables_index()
        elif target == "all":
            build_variables_index()
            print("\n")
            build_docs_index()
        else:
            print(f"Unknown target: {target}")
            print("Usage: python index.py [variables|docs|all]")
    else:
        # Default: build both
        print("Building both indices...\n")
        build_variables_index()
        print("\n")
        build_docs_index()

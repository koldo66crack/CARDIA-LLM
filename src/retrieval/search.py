"""
Shared search utilities for CARDIA RAG retrieval.
Generic functions that work with any FAISS index and metadata.
"""

import os
import pickle
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional

# Cache for loaded models to avoid reloading
_model_cache: Dict[str, SentenceTransformer] = {}


def load_index(index_dir: str, index_name: str = "faiss_index.bin") -> Tuple[Optional[faiss.Index], Optional[List[Dict]], Optional[SentenceTransformer]]:
    """
    Load FAISS index, metadata, and embedding model from a directory.
    
    Args:
        index_dir: Directory containing index files
        index_name: Name of the FAISS index file (default: "faiss_index.bin")
        
    Returns:
        Tuple of (index, metadata, model) or (None, None, None) if not found
    """
    index_path = os.path.join(index_dir, index_name)
    
    # Derive metadata and info paths from index name
    base_name = index_name.replace(".bin", "")
    if base_name == "faiss_index":
        # Default naming convention
        metadata_path = os.path.join(index_dir, "metadata.pkl")
        info_path = os.path.join(index_dir, "index_info.json")
    else:
        # Custom naming (e.g., docs_faiss_index.bin -> docs_metadata.pkl)
        metadata_path = os.path.join(index_dir, f"{base_name.replace('_faiss_index', '')}_metadata.pkl")
        info_path = os.path.join(index_dir, f"{base_name.replace('_faiss_index', '')}_index_info.json")
    
    if not os.path.exists(index_path):
        return None, None, None
    
    if not os.path.exists(metadata_path):
        return None, None, None
    
    # Load FAISS index
    index = faiss.read_index(index_path)
    
    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Load model info and get model
    model_name = 'BAAI/bge-small-en-v1.5'  # Default
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
        model_name = info.get('model_name', model_name)
    
    # Use cached model if available
    if model_name in _model_cache:
        model = _model_cache[model_name]
    else:
        model = SentenceTransformer(model_name)
        _model_cache[model_name] = model
    
    return index, metadata, model


def semantic_search(
    query: str,
    index: faiss.Index,
    metadata: List[Dict],
    model: SentenceTransformer,
    k: int = 100,
    threshold: float = 0.7
) -> List[Dict]:
    """
    Perform semantic search on a FAISS index.
    
    Args:
        query: Search query string
        index: FAISS index to search
        metadata: List of metadata dicts corresponding to index vectors
        model: SentenceTransformer model for embedding the query
        k: Maximum number of results to return
        threshold: Minimum similarity score (0-1) for inclusion
        
    Returns:
        List of matching chunks with 'similarity_score' field added
    """
    # Embed the query
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)
    
    # Search the index
    scores, indices = index.search(query_embedding, k)
    
    # Filter by threshold and format results
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:  # FAISS returns -1 for empty slots
            continue
        if score >= threshold:
            chunk = metadata[idx].copy()
            chunk['similarity_score'] = float(score)
            results.append(chunk)
    
    return results


def keyword_search(
    keywords: List[str],
    metadata: List[Dict],
    fields: List[str]
) -> List[Dict]:
    """
    Perform case-insensitive substring keyword search across specified fields.
    
    Args:
        keywords: List of keywords/tags to search for
        metadata: List of metadata dicts to search through
        fields: List of field names to search in (e.g., ['variable_name', 'dataset'])
        
    Returns:
        List of matching chunks with 'match_type', 'matched_tag', and 'keyword_score' fields
    """
    if not keywords:
        return []
    
    results = []
    seen_indices = set()
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        
        for idx, chunk in enumerate(metadata):
            if idx in seen_indices:
                continue
            
            # Check each field for a match
            match_field = None
            for field in fields:
                field_value = str(chunk.get(field, '')).lower()
                if keyword_lower in field_value:
                    match_field = field
                    break
            
            if match_field:
                chunk_copy = chunk.copy()
                chunk_copy['match_type'] = match_field
                chunk_copy['matched_tag'] = keyword
                chunk_copy['keyword_score'] = 1.0
                results.append(chunk_copy)
                seen_indices.add(idx)
    
    return results


def merge_results(
    keyword_results: List[Dict],
    semantic_results: List[Dict],
    id_field: str = 'id'
) -> List[Dict]:
    """
    Merge keyword and semantic search results, removing duplicates.
    Keyword matches are prioritized (appear first).
    
    Args:
        keyword_results: Results from keyword search
        semantic_results: Results from semantic search
        id_field: Field name to use for deduplication (default: 'id')
        
    Returns:
        Combined list with keyword results first, then unique semantic results
    """
    merged = []
    seen_ids = set()
    
    # Add keyword results first (higher priority)
    for chunk in keyword_results:
        chunk_id = chunk.get(id_field)
        if chunk_id not in seen_ids:
            merged.append(chunk)
            seen_ids.add(chunk_id)
    
    # Add semantic results that aren't duplicates
    for chunk in semantic_results:
        chunk_id = chunk.get(id_field)
        if chunk_id not in seen_ids:
            merged.append(chunk)
            seen_ids.add(chunk_id)
    
    return merged


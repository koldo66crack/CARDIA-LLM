"""
RAG Retriever for CARDIA data dictionary.
Handles semantic search and retrieval of relevant variable chunks.
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime
from collections import defaultdict


def load_existing_index(index_dir="data/processed", verbose=False):
    """
    Load existing FAISS index and metadata.
    
    Args:
        index_dir (str): Directory containing index files
        verbose (bool): Whether to print detailed output
        
    Returns:
        tuple: (index, metadata, model) or (None, None, None) if not found
    """
    index_path = os.path.join(index_dir, "faiss_index.bin")
    metadata_path = os.path.join(index_dir, "metadata.pkl")
    info_path = os.path.join(index_dir, "index_info.json")
    
    if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
        if verbose:
            print("No existing index found")
        return None, None, None
    
    try:
        if verbose:
            print("Loading existing index...")
        
        # Load FAISS index
        index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Load model info
        import json
        with open(info_path, 'r') as f:
            info = json.load(f)
        model_name = info.get('model_name', 'BAAI/bge-small-en-v1.5')
        
        # Load the embedding model
        model = SentenceTransformer(model_name)
        
        if verbose:
            print(f"Loaded index with {index.ntotal} vectors and {len(metadata)} metadata entries")
        return index, metadata, model
        
    except Exception as e:
        if verbose:
            print(f"Error loading index: {e}")
        return None, None, None


def search_index(index, metadata, model, query, k, similarity_threshold=0.7):
    """
    Search the FAISS index for similar chunks.
    
    Args:
        index (faiss.Index): FAISS index
        metadata (list): List of metadata chunks
        model (SentenceTransformer): Embedding model
        query (str): Search query
        k (int): Number of results to return
        similarity_threshold (float): Minimum similarity score
        
    Returns:
        list: List of (chunk, similarity_score) tuples
    """
    # Embed the query
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)
    
    # Search the index
    scores, indices = index.search(query_embedding, k)
    
    # Filter by similarity threshold and format results
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if score >= similarity_threshold:
            chunk = metadata[idx].copy()
            chunk['similarity_score'] = float(score)
            results.append(chunk)
    
    return results


def search_by_tags(tags: List[str], metadata: List[Dict]) -> List[Dict]:
    """
    Search for chunks matching extracted tags (case-insensitive substring match).
    Searches across variable_name, dataset, and label fields.
    
    Args:
        tags (List[str]): Extracted tags/identifiers to search for (e.g., ["aachem", "AL3CREAT", "creatinine"])
        metadata (List[Dict]): All available metadata chunks
        
    Returns:
        list: List of matching chunks with 'match_type' field indicating where match occurred
    """
    if not tags:
        return []
    
    results = []
    seen_indices = set()  # Avoid duplicates if multiple tags match same chunk
    
    for tag in tags:
        tag_lower = tag.lower()
        for idx, chunk in enumerate(metadata):
            if idx in seen_indices:
                continue
            
            # Check if tag matches in variable_name, dataset, or label (case-insensitive substring)
            variable_name = chunk.get('variable_name', '').lower()
            dataset = chunk.get('dataset', '').lower()
            label = chunk.get('label', '').lower()
            
            match_type = None
            if tag_lower in variable_name:
                match_type = 'variable_name'
            elif tag_lower in dataset:
                match_type = 'dataset'
            elif tag_lower in label:
                match_type = 'label'
            
            if match_type:
                chunk_copy = chunk.copy()
                chunk_copy['match_type'] = match_type
                chunk_copy['matched_tag'] = tag
                chunk_copy['keyword_score'] = 1.0
                results.append(chunk_copy)
                seen_indices.add(idx)
    
    return results


def search_variables(query, k=100, similarity_threshold=0.7, index_dir="data/processed", verbose=False):
    """
    Search for variables matching the user query.
    
    Args:
        query (str): User's search query
        k (int): Number of results to return
        similarity_threshold (float): Minimum similarity score
        index_dir (str): Directory containing index files
        verbose (bool): Whether to print detailed output
        
    Returns:
        list: List of matching variable chunks with scores
    """
    if verbose:
        print(f"Searching for: '{query}'")
        print(f"Parameters: k={k}, threshold={similarity_threshold}")
    
    # Load index and metadata
    index, metadata, model = load_existing_index(index_dir, verbose=verbose)
    
    if index is None:
        if verbose:
            print("Error: Could not load index. Please run index.py first.")
        return []
    
    # Search for similar chunks
    results = search_index(index, metadata, model, query, k, similarity_threshold)
    
    if verbose:
        print(f"Found {len(results)} relevant variables")
        
        # Display results summary
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.get('variable_name', 'Unknown')} "
                  f"({result.get('dataset', 'Unknown dataset')}) "
                  f"- Score: {result.get('similarity_score', 0):.3f}")
            if result.get('label'):
                print(f"   Description: {result['label']}")
    
    # Save retrieved chunks to JSON for debugging
    debug_log = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "threshold": similarity_threshold,
        "num_retrieved": len(results),
        "chunks": results
    }
    os.makedirs("data/debug", exist_ok=True)
    with open("data/debug/retrieved_chunks.json", "w", encoding="utf-8") as f:
        json.dump(debug_log, f, indent=2, ensure_ascii=False)
    
    return results




def build_retrieved_context(chunks: List[Dict], user_query: str) -> str:
    """
    Build context string from retrieved chunks and user query.
    
    Args:
        chunks (List[Dict]): Retrieved variable chunks with high similarity to the user query
        user_query (str): User's original query
        
    Returns:
        str: Formatted context for the LLM
    """
    # Build context with retrieved variables
    context_parts = [f"User Query: {user_query}\n"]
    context_parts.append("Relevant Variables from CARDIA Data Dictionary, retrieved with high similarity to the user query:")
    context_parts.append("=" * 50)
    
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"\n{i}. Variable: {chunk.get('variable_name', 'Unknown')}")
        context_parts.append(f"   Dataset: {chunk.get('dataset', 'Unknown')}")
        
        if chunk.get('label'):
            context_parts.append(f"   Description: {chunk['label']}")
        
        context_parts.append(f"   Type: {chunk.get('type', 'Unknown')}")
        
        if chunk.get('length'):
            context_parts.append(f"   Length: {chunk['length']}")
        
        if chunk.get('number_observations'):
            context_parts.append(f"   Observations: {chunk['number_observations']}")
        
        if chunk.get('format'):
            context_parts.append(f"   Format: {chunk['format']}")
        
        # Add similarity score for reference
        if chunk.get('similarity_score'):
            context_parts.append(f"   Relevance Score: {chunk['similarity_score']:.3f}")

    context_parts.append("Your response:")
    
    return "\n".join(context_parts)


if __name__ == "__main__":
    # Example usage
    print("CARDIA RAG Retriever")
    print("=" * 40)
    
    # Test basic search
    query = "blood pressure"
    print(f"\nTesting search for: '{query}'")
    results = search_variables(query, k=3, verbose=True)
    
    if results:
        print(f"\nTop result: {results[0]['variable_name']}")
        print(f"Description: {results[0].get('label', 'No description')}")

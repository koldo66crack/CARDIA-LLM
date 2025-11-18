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


def search_variables(query, k=200, similarity_threshold=0.7, index_dir="data/processed", verbose=False):
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

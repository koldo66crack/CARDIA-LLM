"""
Variable retriever for CARDIA RAG system.
Handles hybrid search (keyword + semantic) on the BIOLINCC variable index.
"""

from typing import List, Dict, Optional
from .search import load_index, semantic_search, keyword_search, merge_results

# Index configuration
INDEX_DIR = "data/processed"
INDEX_NAME = "faiss_index.bin"  # Default naming for variables

# Fields to search for keyword matching
KEYWORD_FIELDS = ['variable_name', 'dataset', 'label']


def search_variables(
    query: str,
    keywords: Optional[List[str]] = None,
    k: int = 100,
    threshold: float = 0.75
) -> List[Dict]:
    """
    Search for variables using hybrid approach (keyword + semantic).
    
    Args:
        query: Optimized search query for semantic search
        keywords: List of keywords/tags for exact matching (e.g., dataset names, variable names)
        k: Maximum number of semantic results to retrieve
        threshold: Minimum similarity score for semantic results
        
    Returns:
        List of variable chunks with search metadata.
        Keyword matches appear first, followed by unique semantic matches.
        Each chunk includes: variable_name, dataset, label, study, type, etc.
    """
    keywords = keywords or []
    
    # Load index
    index, metadata, model = load_index(INDEX_DIR, INDEX_NAME)
    
    if index is None:
        print("Error: Variable index not found. Please run index.py first.")
        return []
    
    # Keyword search
    keyword_results = keyword_search(keywords, metadata, KEYWORD_FIELDS)
    
    # Semantic search
    semantic_results = semantic_search(query, index, metadata, model, k=k, threshold=threshold)
    
    # Merge results (keyword first, then semantic, deduplicated)
    results = merge_results(keyword_results, semantic_results, id_field='id')
    
    return results


def get_datasets_from_results(results: List[Dict]) -> List[str]:
    """
    Extract unique dataset names from search results.
    Useful for enriching doc search keywords.
    
    Args:
        results: List of variable chunks from search
        
    Returns:
        List of unique dataset names
    """
    datasets = set()
    for chunk in results:
        dataset = chunk.get('dataset')
        if dataset:
            datasets.add(dataset)
    return list(datasets)


def get_variable_names_from_results(results: List[Dict]) -> List[str]:
    """
    Extract unique variable names from search results.
    
    Args:
        results: List of variable chunks from search
        
    Returns:
        List of unique variable names
    """
    names = set()
    for chunk in results:
        name = chunk.get('variable_name')
        if name:
            names.add(name)
    return list(names)


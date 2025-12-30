"""
Documentation retriever for CARDIA RAG system.
Handles hybrid search (keyword + semantic) on the CARDIA documentation index.
"""

from typing import List, Dict, Optional
from .search import load_index, semantic_search, keyword_search, merge_results

# Index configuration
INDEX_DIR = "data/processed"
INDEX_NAME = "docs_faiss_index.bin"

# Fields to search for keyword matching (filename contains dataset name)
KEYWORD_FIELDS = ['filename']


def search_docs(
    query: str,
    dataset_names: Optional[List[str]] = None,
    k: int = 100,
    threshold: float = 0.70
) -> List[Dict]:
    """
    Search for documentation pages using hybrid approach (keyword + semantic).
    
    Args:
        query: Optimized search query for semantic search
        dataset_names: List of dataset names for keyword matching against filenames
                      (e.g., ["CAF38", "aachem"] will match "Caf38.pdf", "aachem.pdf")
        k: Maximum number of semantic results to retrieve
        threshold: Minimum similarity score for semantic results
        
    Returns:
        List of doc chunks with search metadata.
        Keyword matches appear first, followed by unique semantic matches.
        Each chunk includes: filename, wave, year, page, content, doc_type, etc.
    """
    dataset_names = dataset_names or []
    
    # Load index
    index, metadata, model = load_index(INDEX_DIR, INDEX_NAME)
    
    if index is None:
        print("Warning: Documentation index not found. Run index.py to build docs index.")
        return []
    
    # Keyword search (match dataset names against filenames)
    keyword_results = keyword_search(dataset_names, metadata, KEYWORD_FIELDS)
    
    # Semantic search
    semantic_results = semantic_search(query, index, metadata, model, k=k, threshold=threshold)
    
    # Merge results (keyword first, then semantic, deduplicated)
    results = merge_results(keyword_results, semantic_results, id_field='id')
    
    return results


def get_unique_sources(results: List[Dict]) -> List[str]:
    """
    Extract unique source files from search results.
    Useful for citing sources in responses.
    
    Args:
        results: List of doc chunks from search
        
    Returns:
        List of unique source file paths
    """
    sources = set()
    for chunk in results:
        source = chunk.get('source_file')
        if source:
            sources.add(source)
    return list(sources)


def group_by_source(results: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group doc chunks by their source file.
    Useful for understanding which documents were retrieved.
    
    Args:
        results: List of doc chunks from search
        
    Returns:
        Dict mapping source_file -> list of chunks from that file
    """
    grouped = {}
    for chunk in results:
        source = chunk.get('source_file', 'unknown')
        if source not in grouped:
            grouped[source] = []
        grouped[source].append(chunk)
    return grouped


"""
Context builder for CARDIA RAG system.
Builds formatted context strings from retrieved variable and documentation chunks.
"""

from typing import List, Dict


def build_context(variable_chunks: List[Dict], doc_chunks: List[Dict]) -> str:
    """
    Build unified context string from variable and documentation chunks.
    
    Args:
        variable_chunks: Retrieved variable chunks from data dictionary
        doc_chunks: Retrieved documentation chunks from PDFs/Word docs
        
    Returns:
        Formatted context string for LLM consumption
    """
    parts = []
    
    # Variable section
    if variable_chunks:
        parts.append("=" * 60)
        parts.append("CARDIA VARIABLES (from Data Dictionary)")
        parts.append("=" * 60)
        for i, chunk in enumerate(variable_chunks, 1):
            parts.append(format_variable_chunk(chunk, i))
    
    # Documentation section
    if doc_chunks:
        if variable_chunks:
            parts.append("")  # Add spacing between sections
        parts.append("=" * 60)
        parts.append("CARDIA DOCUMENTATION (from Study Documents)")
        parts.append("=" * 60)
        for i, chunk in enumerate(doc_chunks, 1):
            parts.append(format_doc_chunk(chunk, i))
    
    return "\n".join(parts)


def format_variable_chunk(chunk: Dict, index: int) -> str:
    """
    Format a single variable chunk for LLM context.
    
    Args:
        chunk: Variable chunk dictionary
        index: Display index (1-based)
        
    Returns:
        Formatted string representation
    """
    lines = [
        f"\n{index}. Variable: {chunk.get('variable_name', 'Unknown')}",
        f"   Dataset: {chunk.get('dataset', 'Unknown')}",
        f"   Study: {chunk.get('study', 'unknown').capitalize()}",
        f"   Description: {chunk.get('label', '')}",
        f"   Type: {chunk.get('type', 'Unknown')}",
        f"   Length: {chunk.get('length', 'Unknown')}",
        f"   Observations: {chunk.get('number_observations', 'Unknown')}",
        f"   Format: {chunk.get('format', '')}",
    ]
    
    # Search metadata (these are conditionally added during retrieval)
    if chunk.get('similarity_score'):
        lines.append(f"   Relevance: {chunk['similarity_score']:.3f}")
    elif chunk.get('keyword_score'):
        lines.append(f"   Match: keyword ({chunk.get('match_type', 'exact')})")
    
    return "\n".join(lines)


def format_doc_chunk(chunk: Dict, index: int) -> str:
    """
    Format a single documentation chunk for LLM context.
    
    Args:
        chunk: Documentation chunk dictionary
        index: Display index (1-based)
        
    Returns:
        Formatted string representation
    """
    year = chunk.get('year', -1)
    
    lines = [f"\n{index}. Source: {chunk.get('filename', 'Unknown')}"]
    
    # Wave and year info (only show year if valid)
    if year != -1:
        lines.append(f"   Wave: {chunk.get('wave', 'Unknown')} (Year {year})")
    else:
        lines.append(f"   Wave: {chunk.get('wave', 'Unknown')}")
    
    lines.append(f"   Document Type: {chunk.get('doc_type', 'Unknown')}")
    lines.append(f"   Page: {chunk.get('page', 'Unknown')}")
    lines.append(f"   Content:\n{chunk.get('content', '')}")
    
    # Search metadata
    if chunk.get('similarity_score'):
        lines.append(f"   Relevance: {chunk['similarity_score']:.3f}")
    elif chunk.get('keyword_score'):
        lines.append(f"   Match: keyword ({chunk.get('match_type', 'filename')})")
    
    return "\n".join(lines)

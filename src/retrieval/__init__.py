"""
Retrieval module for CARDIA RAG system.
Provides shared search utilities and specialized retrievers for variables and documentation.
"""

from .search import load_index, semantic_search, keyword_search, merge_results
from .variables import search_variables, get_datasets_from_results, get_variable_names_from_results


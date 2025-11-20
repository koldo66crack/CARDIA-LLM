"""
Quick test script to verify hybrid search implementation.
Run this to test keyword matching without needing to run the full app.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rag_retriever import load_existing_index, search_by_tags

def test_keyword_matching():
    """Test the keyword matching function"""
    print("=" * 70)
    print("Testing Keyword Matching")
    print("=" * 70)
    
    # Load metadata
    print("\n1. Loading metadata...")
    index, metadata, model = load_existing_index(verbose=True)
    
    if index is None:
        print("Error: Could not load index. Please run index.py first.")
        return
    
    # Test 1: Single dataset tag
    print("\n" + "-" * 70)
    print("Test 1: Single dataset tag")
    print("-" * 70)
    tags = ["aachem"]
    matches = search_by_tags(tags, metadata)
    print(f"Query tags: {tags}")
    print(f"Matches found: {len(matches)}")
    if matches:
        print("Sample results:")
        for i, match in enumerate(matches[:5]):
            print(f"  {i+1}. {match.get('variable_name', 'Unknown')} "
                  f"({match.get('dataset', 'Unknown')})")
            print(f"     Match: {match.get('match_type')} = {match.get('matched_tag')}")
    
    # Test 2: Variable code tag
    print("\n" + "-" * 70)
    print("Test 2: Variable code tag")
    print("-" * 70)
    tags = ["AL3CREAT"]
    matches = search_by_tags(tags, metadata)
    print(f"Query tags: {tags}")
    print(f"Matches found: {len(matches)}")
    if matches:
        print("Results:")
        for i, match in enumerate(matches):
            print(f"  {i+1}. {match.get('variable_name', 'Unknown')} "
                  f"in {match.get('dataset', 'Unknown')}")
            print(f"     Label: {match.get('label', 'N/A')}")
            print(f"     Match: {match.get('match_type')} = {match.get('matched_tag')}")
    
    # Test 3: Multiple tags
    print("\n" + "-" * 70)
    print("Test 3: Multiple tags")
    print("-" * 70)
    tags = ["aachem", "AL3CREAT"]
    matches = search_by_tags(tags, metadata)
    print(f"Query tags: {tags}")
    print(f"Matches found: {len(matches)} (should be ~all aachem variables)")
    print("This ensures AL3CREAT in aachem isn't duplicated")
    
    # Test 4: Case insensitivity
    print("\n" + "-" * 70)
    print("Test 4: Case insensitivity")
    print("-" * 70)
    tags_uppercase = ["AACHEM"]
    tags_lowercase = ["aachem"]
    tags_mixedcase = ["AaCheM"]
    
    matches_upper = search_by_tags(tags_uppercase, metadata)
    matches_lower = search_by_tags(tags_lowercase, metadata)
    matches_mixed = search_by_tags(tags_mixedcase, metadata)
    
    print(f"AACHEM (uppercase): {len(matches_upper)} matches")
    print(f"aachem (lowercase): {len(matches_lower)} matches")
    print(f"AaCheM (mixedcase): {len(matches_mixed)} matches")
    print(f"All equal? {len(matches_upper) == len(matches_lower) == len(matches_mixed)}")
    
    # Test 5: Substring matching
    print("\n" + "-" * 70)
    print("Test 5: Substring matching")
    print("-" * 70)
    tags = ["AL3CR"]  # Substring of AL3CREAT
    matches = search_by_tags(tags, metadata)
    print(f"Query tags: {tags} (substring)")
    print(f"Matches found: {len(matches)}")
    if matches:
        print("Results (should include AL3CREAT and similar):")
        for i, match in enumerate(matches[:10]):
            print(f"  {i+1}. {match.get('variable_name', 'Unknown')}")
    
    # Test 6: No matches
    print("\n" + "-" * 70)
    print("Test 6: No matches")
    print("-" * 70)
    tags = ["NONEXISTENT123"]
    matches = search_by_tags(tags, metadata)
    print(f"Query tags: {tags}")
    print(f"Matches found: {len(matches)} (expected: 0)")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    test_keyword_matching()


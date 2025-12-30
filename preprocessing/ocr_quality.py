"""
OCR Quality Checks for CARDIA PDF Extraction.

Determines when OCR output is broken and needs VLM fallback.
Heuristics based on empirical testing with CARDIA documentation.
"""

import re


def should_use_vlm(text: str) -> bool:
    """
    Determine if OCR output needs VLM fallback for complex tables.
    
    Triggers VLM if EITHER:
    - Too many short lines (>15%) - indicates broken table columns
    - Table detected (5+ "Num"/"Char") - tables often need special handling
    
    Note: Low OCR confidence is checked separately before calling this function.
    
    Args:
        text: OCR extracted text
        
    Returns:
        bool: True if VLM should be used
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return True
    
    # Check 1: Short lines (broken columns)
    short_lines = 0
    for line in lines:
        # Skip lines ending in punctuation (likely code/sentences)
        if line.endswith(';') or line.endswith(',') or line.endswith('.'):
            continue
        
        word_count = len(line.split())
        if word_count > 0 and word_count <= 2:
            short_lines += 1
    
    short_ratio = short_lines / len(lines)
    if short_ratio > 0.15:
        return True
    
    # Check 2: Table detection (5+ "Num" or "Char")
    num_count = len(re.findall(r'\bNum\b', text, re.IGNORECASE))
    char_count = len(re.findall(r'\bChar\b', text, re.IGNORECASE))
    total = num_count + char_count
    
    if total >= 5:
        return True
    
    return False


if __name__ == "__main__":
    # Test with sample text
    test_broken_table = """
    lable Type Len Pos Format Label
    ALCOHOL (GM)
    ASTT (CM)
    SoH
    [oe
    8
    8
    8 200 ALPHA TOCOPHEROL
    """
    
    test_good = """
    This is a normal paragraph of text that spans multiple words.
    It contains complete sentences with proper structure.
    The OCR has extracted this text correctly without breaking columns.
    We would like to ask you a few questions about yourself.
    In the past year, have you been a patient in a hospital overnight?
    """
    
    print("Testing broken table OCR (short lines):")
    print(f"  Use VLM: {should_use_vlm(test_broken_table)}")
    
    print("\nTesting good OCR:")
    print(f"  Use VLM: {should_use_vlm(test_good)}")
    
    print("\nNote: Low-quality scans are now detected via OCR confidence scores,")
    print("checked before this function is called.")


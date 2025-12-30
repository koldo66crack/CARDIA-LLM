"""
VLM (Vision Language Model) utilities for PDF text extraction.

Uses Gemini Vision to extract text from complex PDF pages (tables, diagrams).
Supports multiple API keys with automatic rotation to manage rate limits.
"""

import os
import io
import base64
import random
import time
from pathlib import Path
from threading import Lock
from dotenv import load_dotenv
import google.generativeai as genai
from pdf2image import convert_from_path

load_dotenv()

# Thread lock for key selection
_quota_lock = Lock()


def load_api_keys() -> list[str]:
    """Load all available Gemini API keys from environment."""
    keys = []
    for i in range(1, 6):  # GEMINI_API_KEY_1 through GEMINI_API_KEY_4
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if key:
            keys.append(key)
    
    if not keys:
        raise ValueError("No GEMINI_API_KEY_* found in environment variables")
    
    return keys


def get_next_key() -> str:
    """
    Get a random API key for load distribution across keys.
    
    Uses random selection instead of round-robin to work correctly
    with parallel processing where each worker is a separate process.
    
    Returns:
        str: API key to use
    """
    keys = load_api_keys()
    return random.choice(keys)


def load_vlm_prompt() -> str:
    """Load the VLM extraction prompt from file."""
    prompt_path = Path(__file__).parent / "vlm_extraction_prompt.txt"
    
    if prompt_path.exists():
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Fallback prompt if file not found
    return """Extract all text from this document page.

Instructions:
- Preserve table structure using | for column separators
- Maintain proper row alignment
- Include all text, numbers, and labels
- For diagrams or questionnaires, describe the structure in natural language
- Return only the extracted content, no commentary

Output the extracted text:"""


def image_to_base64(image) -> str:
    """Convert PIL Image to base64 string for API."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def extract_with_vlm(image, model_name: str = "gemini-2.0-flash-exp", max_retries: int = 5) -> str:
    """
    Extract text from an image using Gemini Vision with API key rotation.
    
    Retries with different keys on any error. After max_retries, raises error
    to stop all processing.
    
    Args:
        image: PIL Image object
        model_name: Gemini model to use
        max_retries: Number of retry attempts (default: 5)
        
    Returns:
        str: Extracted text from the image
        
    Raises:
        RuntimeError: If all retries exhausted
    """
    prompt = load_vlm_prompt()
    last_error = None
    
    for attempt in range(max_retries):
        # Get random API key (naturally avoids the one that just failed)
        api_key = get_next_key()
        key_suffix = api_key[-5:] if len(api_key) >= 5 else "****"
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        try:
            # Send image to Gemini Vision
            response = model.generate_content([prompt, image])
            return response.text
            
        except Exception as e:
            last_error = e
            
            if attempt < max_retries - 1:
                # Not the last attempt - retry with cooldown
                print(f"\n  [VLM error on key ...{key_suffix}] Waiting 10s and retrying with different key (attempt {attempt + 1}/{max_retries})...")
                time.sleep(10)
                continue
            else:
                # All retries exhausted
                raise RuntimeError(
                    f"VLM API error on key ...{key_suffix} after {max_retries} attempts. "
                    f"Last error: {str(last_error)}. Processing stopped."
                )


def extract_page_with_vlm(pdf_path: str, page_num: int, model_name: str = "gemini-2.0-flash-exp") -> str:
    """
    Extract text from a specific PDF page using VLM.
    
    Args:
        pdf_path: Path to the PDF file
        page_num: Page number (1-indexed)
        model_name: Gemini model to use
        
    Returns:
        str: Extracted text from the page
    """
    # Convert PDF page to image
    images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
    
    if not images:
        raise ValueError(f"Could not convert page {page_num} to image")
    
    return extract_with_vlm(images[0], model_name)


if __name__ == "__main__":
    # Show API key configuration
    print("VLM Configuration")
    print("=" * 60)
    try:
        num_keys = len(load_api_keys())
        print(f"API keys configured: {num_keys}")
        print("Keys will rotate automatically to distribute load")
    except Exception as e:
        print(f"Error loading keys: {e}")
    
    print("\n" + "=" * 60)
    print("Testing VLM extraction...")
    print("=" * 60)
    
    # Test VLM extraction on a sample page
    sample_pdf = "data/raw/CARDIA documentation/Y00/DOC/AAF06A.PDF"
    
    try:
        text = extract_page_with_vlm(sample_pdf, page_num=1)
        print(f"Extracted text ({len(text)} chars):")
        print("-" * 60)
        print(text[:500] + "..." if len(text) > 500 else text)
        print("\n" + "=" * 60)
        print("Test successful!")
    except RuntimeError as e:
        print(f"Quota limit reached: {e}")
    except Exception as e:
        print(f"Error: {e}")


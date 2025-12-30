"""
Query optimization and sufficiency checking for CARDIA RAG system.
Handles query enhancement and determines when documentation search is needed.
"""

import os
import json
import re
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

from src.context_builder import build_context
from src.retrieval.variables import get_datasets_from_results

load_dotenv()

# Prompt file paths
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
QUERY_OPTIMIZATION_PROMPT = os.path.join(os.path.dirname(__file__), "rag_query_optimization_prompt.txt")
SUFFICIENCY_CHECK_PROMPT = os.path.join(PROMPTS_DIR, "sufficiency_check.txt")


def _get_openai_client() -> OpenAI:
    """Get configured OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    return OpenAI(api_key=api_key)


def _extract_json_from_response(response_text: str) -> Dict:
    """
    Extract JSON from LLM response, handling markdown code blocks and chain-of-thought.
    
    Args:
        response_text: Response text that may contain markdown code blocks or reasoning
        
    Returns:
        Parsed JSON content
    """
    text = response_text.strip()
    
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()
    else:
        # Try to find raw JSON object (last occurrence for CoT responses)
        json_matches = list(re.finditer(r'\{[^{}]*\}', text, re.DOTALL))
        if json_matches:
            text = json_matches[-1].group()
    
    return json.loads(text)


def _format_conversation_history(history: List[Dict[str, str]]) -> str:
    """Format conversation history for prompt inclusion."""
    if not history:
        return ""
    
    formatted = "Previous conversation:\n"
    for msg in history[-4:]:  # Last 2 exchanges
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted += f"{role}: {msg['content']}\n"
    return formatted + "\n"


def optimize_query(user_query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict:
    """
    Generate an optimized RAG search query and extract keywords using GPT-4o-mini.
    
    Args:
        user_query: User's current question
        conversation_history: Previous conversation exchanges
        
    Returns:
        Dict with "query" (optimized search query) and "tags" (extracted keywords)
    """
    client = _get_openai_client()
    
    # Load prompt template
    with open(QUERY_OPTIMIZATION_PROMPT, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    # Format prompt
    history_text = _format_conversation_history(conversation_history or [])
    prompt = prompt_template.format(
        history_section=history_text,
        user_query=user_query
    )
    
    # Call LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.25
    )
    
    # Parse response
    try:
        result = _extract_json_from_response(response.choices[0].message.content)
        return {
            "query": result.get("query", user_query),
            "tags": result.get("tags", [])
        }
    except (json.JSONDecodeError, KeyError):
        print(f"Warning: Failed to parse query optimization response")
        return {"query": user_query, "tags": []}


def check_sufficiency(user_query: str, variable_chunks: List[Dict]) -> Dict:
    """
    Check if retrieved variables are sufficient to answer the user's question.
    Uses chain-of-thought reasoning before making a decision.
    
    Args:
        user_query: Original user question
        variable_chunks: Retrieved variable chunks from Stage 1
        
    Returns:
        Dict with:
        - sufficient: bool - whether variables alone can answer the question
        - reason: str - explanation of the decision
        - doc_query: str - optimized query for doc search (empty if sufficient)
        - datasets: List[str] - unique dataset names from variable chunks (for keyword matching)
    """
    client = _get_openai_client()
    
    # Load prompt template
    with open(SUFFICIENCY_CHECK_PROMPT, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    # Format variables using context builder
    formatted_variables = build_context(variable_chunks, [])
    if not formatted_variables:
        formatted_variables = "No variables were retrieved."
    
    # Format prompt
    prompt = prompt_template.format(
        variable_summary=formatted_variables,
        user_query=user_query
    )
    
    # Call LLM with higher max_tokens to allow reasoning
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.25,
        max_tokens=2000
    )
    
    # Get datasets
    datasets = get_datasets_from_results(variable_chunks)
    
    # Parse response
    try:
        result = _extract_json_from_response(response.choices[0].message.content)
        return {
            "sufficient": result.get("sufficient", True),
            "reason": result.get("reason", ""),
            "doc_query": result.get("doc_query", ""),
            "datasets": datasets
        }
    except (json.JSONDecodeError, KeyError):
        print(f"Warning: Failed to parse sufficiency check response")
        # Default to sufficient if parsing fails (avoid unnecessary doc search)
        return {
            "sufficient": True,
            "reason": "Failed to parse LLM response",
            "doc_query": "",
            "datasets": datasets
        }

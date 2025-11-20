"""
Main RAG query flow: user query → retrieval → LLM generation
Orchestrates the complete retrieval-augmented generation pipeline.
"""

import os
import json
import re
from typing import Optional, List, Dict
from dotenv import load_dotenv
import google.generativeai as genai
from rag_retriever import search_variables, build_retrieved_context
from conversation_manager import ChatSession

# Load environment variables from .env file
load_dotenv()


def extract_json_from_response(response_text: str) -> Dict:
    """
    Extract JSON from LLM response, handling markdown code blocks.
    
    Args:
        response_text (str): Response text that may contain markdown code blocks
        
    Returns:
        dict: Parsed JSON content
    """
    text = response_text.strip()
    
    # Try to extract JSON from markdown code blocks
    # Matches ```json ... ``` or ``` ... ```
    json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()
    
    # Parse and return JSON
    return json.loads(text)


def generate_rag_query(user_query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, any]:
    """
    Generate an optimized RAG search query and extract tags using Gemini 2.0 Flash.
    Uses isolated generate_content() call (not part of conversation history).
    
    Returns both the optimized query and extracted tags for keyword matching.
    Always calls LLM to extract tags, but only includes conversation history for follow-ups.
    
    Args:
        user_query (str): User's current question
        conversation_history (Optional[List[Dict[str, str]]]): Previous conversation exchanges with roles
        
    Returns:
        dict: {"query": str, "tags": List[str]}
              - query: Optimized search query for semantic retrieval
              - tags: Extracted identifiers for keyword matching
    """
    # Configure Gemini with API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in .env file.")
    genai.configure(api_key=api_key)
    
    # Create model for isolated query optimization
    model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")
    
    # Load the optimization prompt template
    prompt_template_path = os.path.join(os.path.dirname(__file__), "rag_query_optimization_prompt.txt")
    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    # Format conversation history with roles (only for follow-up messages)
    history_text = ""
    if conversation_history and len(conversation_history) > 0:
        history_text = "Previous conversation:\n"
        for msg in conversation_history[-4:]:  # Only last 2 exchanges to keep context focused
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n" 
        history_text += "\n"
    
    # Format the prompt
    optimization_prompt = prompt_template.format(
        history_section=history_text,
        user_query=user_query
    )
    
    # Use isolated generate_content call (not part of conversation history)
    response = model.generate_content(optimization_prompt)
    
    # Parse JSON response
    try:
        result = extract_json_from_response(response.text)
        return {
            "query": result.get("query", user_query),
            "tags": result.get("tags", [])
        }
    except (json.JSONDecodeError, KeyError) as e:
        # Fallback if JSON parsing fails
        print(f"Warning: Failed to parse LLM response as JSON. Response was: {response.text}")
        return {"query": user_query, "tags": []}


def generate_response(user_query: str, chat_session: ChatSession, similarity_threshold: float = 0.75) -> str:
    """
    Generate response using hybrid RAG pipeline with semantic + keyword search.
    
    Pipeline:
    1. Generate optimized RAG query and extract tags
    2. Retrieve keyword matches (if tags found)
    3. Retrieve semantically similar chunks
    4. Combine results (keyword matches first, then semantic)
    5. Send to LLM via ChatSession with optional reference material
    
    Args:
        user_query (str): User's question
        chat_session (ChatSession): Active chat session managing conversation history
        similarity_threshold (float): Minimum similarity score for chunk inclusion (default: 0.75)
        
    Returns:
        str: Generated response from the LLM
    """
    try:
        from rag_retriever import search_by_tags, load_existing_index
        
        # Step 1: Generate optimized search query and extract tags
        print(f"Optimizing search query and extracting tags...")
        rag_result = generate_rag_query(user_query, chat_session.get_history())
        rag_query = rag_result["query"]
        tags = rag_result["tags"]
        print(f"Optimized query: {rag_query}")
        if tags:
            print(f"Extracted tags: {tags}")
        else:
            print("No tags found")
        print()
        
        # Step 2: Get all metadata for keyword matching
        index, metadata, model = load_existing_index(verbose=False)
        if index is None:
            print("Error: Could not load index. Proceeding with semantic search only.")
            keyword_matches = []
        else:
            keyword_matches = search_by_tags(tags, metadata)
            if keyword_matches:
                print(f"Found {len(keyword_matches)} keyword matches")
        
        # Step 3: Retrieve semantic matches
        print(f"Searching for semantically similar variables...")
        semantic_matches = search_variables(rag_query, similarity_threshold=similarity_threshold, verbose=False)
        
        # Step 4: Combine results (keyword first, then semantic, avoiding duplicates)
        all_chunks = []
        seen_indices = set()
        
        # Add keyword matches first
        for chunk in keyword_matches:
            # Create a unique identifier for deduplication
            chunk_id = (chunk.get('variable_name'), chunk.get('dataset'))
            if chunk_id not in seen_indices:
                all_chunks.append(chunk)
                seen_indices.add(chunk_id)
        
        # Add semantic matches
        for chunk in semantic_matches:
            chunk_id = (chunk.get('variable_name'), chunk.get('dataset'))
            if chunk_id not in seen_indices:
                all_chunks.append(chunk)
                seen_indices.add(chunk_id)
        
        # Step 5: Format retrieved context if chunks found
        reference_context = None
        if all_chunks:
            reference_context = build_retrieved_context(all_chunks, user_query)
            print(f"Found {len(all_chunks)} total relevant variables "
                  f"({len(keyword_matches)} keyword, {len(all_chunks) - len(keyword_matches)} semantic)")
        else:
            print(f"No relevant variables found. Proceeding without retrieved context.")
        
        # Step 6: Send to LLM with optional reference material
        print("Generating response...")
        response = chat_session.send(user_query, reference_context=reference_context)
        
        return response
        
    except Exception as e:
        return f"Error generating response: {str(e)}"


def main():
    """
    Main function for interactive conversation with RAG-enhanced chatbot.
    """
    print("CARDIA Data Dictionary Assistant")
    print("=" * 50)
    print("Ask me about variables in the CARDIA study!")
    print("Type 'quit' to exit.\n")
    
    # Load system instructions
    instructions_path = os.path.join(os.path.dirname(__file__), "system_instructions.txt")
    with open(instructions_path, 'r', encoding='utf-8') as f:
        system_instructions = f.read()
    
    # Initialize chat session with Gemini
    chat_session = ChatSession(
        provider="gemini",
        model_name="gemini-2.0-flash-exp",
        system_instruction=system_instructions
    )
    
    while True:
        try:
            # Get user query
            user_query = input("\nYour question: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_query:
                continue
            
            # Generate and display response
            response = generate_response(user_query, chat_session, similarity_threshold=0.75)
            print(f"\nResponse:\n{response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()

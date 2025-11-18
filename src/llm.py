"""
Main RAG query flow: user query → retrieval → LLM generation
Orchestrates the complete retrieval-augmented generation pipeline.
"""

import os
from typing import Optional, List, Dict
from dotenv import load_dotenv
import google.generativeai as genai
from rag_retriever import search_variables, build_retrieved_context
from conversation_manager import ChatSession

# Load environment variables from .env file
load_dotenv()


def generate_rag_query(user_query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Generate an optimized RAG search query using Gemini 2.0 Flash.
    Uses isolated generate_content() call (not part of conversation history).
    Incorporates previous conversation context for better query understanding.
    
    Args:
        user_query (str): User's current question
        conversation_history (Optional[List[Dict[str, str]]]): Previous conversation exchanges with roles
        
    Returns:
        str: Optimized search query for semantic retrieval
    """
    # Configure Gemini with API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in .env file.")
    genai.configure(api_key=api_key)
    
    # Create model for isolated query optimization
    model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")
    
    # Format conversation history with roles
    history_text = ""
    if conversation_history:
        history_text = "Previous conversation:\n"
        for msg in conversation_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
        history_text += "\n"
    
    # Prompt to generate optimized search query
    optimization_prompt = f"""{history_text}Given the user's question and any previous context, generate a concise, optimized search query 
that captures the user's intent. The query should be about as long as the user's question.

Current question: {user_query}

Optimized search query:"""
    
    # Use isolated generate_content call (not part of conversation history)
    response = model.generate_content(optimization_prompt)
    
    return response.text.strip()


def generate_response(user_query: str, chat_session: ChatSession, similarity_threshold: float = 0.75) -> str:
    """
    Generate response using RAG pipeline with threshold-based chunk inclusion.
    
    Pipeline:
    1. Generate optimized RAG query from user query
    2. Retrieve semantically similar chunks (filtered by threshold)
    3. Send to LLM via ChatSession with optional reference material
    
    Args:
        user_query (str): User's question
        chat_session (ChatSession): Active chat session managing conversation history
        similarity_threshold (float): Minimum similarity score for chunk inclusion (default: 0.75)
        
    Returns:
        str: Generated response from the LLM
    """
    try:
        # Step 1: Generate optimized search query
        print(f"Optimizing search query...")
        rag_query = generate_rag_query(user_query, chat_session.get_history())
        print(f"Optimized query: {rag_query}\n")
        
        # Step 2: Retrieve relevant chunks
        print(f"Searching for relevant variables...")
        chunks = search_variables(rag_query, k=10, similarity_threshold=similarity_threshold, verbose=False)
        
        # Step 3: Format retrieved context if chunks found
        reference_context = None
        if chunks:
            reference_context = build_retrieved_context(chunks, user_query)
            print(f"Found {len(chunks)} relevant variables above threshold ({similarity_threshold})")
        else:
            print(f"No relevant variables found above threshold ({similarity_threshold}). Proceeding without retrieved context.")
        
        # Step 4: Send to LLM with optional reference material
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

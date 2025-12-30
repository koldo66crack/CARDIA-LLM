"""
Main RAG pipeline for CARDIA LLM.
Orchestrates the two-stage retrieval (variables → docs) and response generation.
"""

from typing import Callable, Optional
from src.query_optimizer import optimize_query, check_sufficiency
from src.retrieval.variables import search_variables
from src.retrieval.docs import search_docs
from src.context_builder import build_context
from src.conversation_manager import ChatSession


def generate_response(
    user_query: str,
    chat_session: ChatSession,
    variable_threshold: float = 0.75,
    doc_threshold: float = 0.50,
    max_variable_chunks: int = 50,
    max_doc_chunks: int = 15,
    status_callback: Optional[Callable[[str], None]] = None
) -> str:
    """
    Generate response using two-stage RAG pipeline.
    
    Pipeline:
    1. Optimize query and extract keywords
    2. Search variables (keyword + semantic)
    3. Check if variables are sufficient to answer
    4. If not sufficient, search documentation
    5. Build context and generate response
    
    Args:
        user_query: User's question
        chat_session: Active chat session for conversation history and generation
        variable_threshold: Minimum similarity score for variable retrieval
        doc_threshold: Minimum similarity score for doc retrieval
        max_variable_chunks: Maximum number of variable chunks to retrieve
        max_doc_chunks: Maximum number of documentation pages to retrieve
        status_callback: Optional callback function for progress updates
        
    Returns:
        Generated response string
    """
    def update_status(message: str):
        if status_callback:
            status_callback(message)
        print(message)
    
    # Stage 1: Optimize query and search variables
    update_status("Searching variables...")
    rag_result = optimize_query(user_query, chat_session.get_history())
    
    variable_chunks = search_variables(
        query=rag_result['query'],
        keywords=rag_result['tags'],
        k=max_variable_chunks,
        threshold=variable_threshold
    )
    print(f"  Found {len(variable_chunks)} relevant variables")
    
    # Stage 2: Check sufficiency
    update_status("Checking if this is enough to answer your question...")
    sufficiency = check_sufficiency(user_query, variable_chunks)
    print(f"  Sufficient: {sufficiency['sufficient']}")
    print(f"  Reason: {sufficiency['reason']}")
    
    # Stage 3: Search documentation if needed
    doc_chunks = []
    if not sufficiency['sufficient']:
        update_status("Searching further in the documentation...")
        
        doc_chunks = search_docs(
            query=sufficiency['doc_query'],
            dataset_names=sufficiency['datasets'],
            k=max_doc_chunks,
            threshold=doc_threshold
        )
        print(f"  Found {len(doc_chunks)} relevant doc pages")
    
    # Stage 4: Build context and generate response
    update_status("Generating response...")
    context = build_context(variable_chunks, doc_chunks)
    
    # Send to LLM with context
    if context:
        response = chat_session.send(user_query, reference_context=context)
    else:
        response = chat_session.send(user_query)
    
    return response


def main():
    """Interactive CLI for testing the pipeline."""
    import os
    
    print("=" * 60)
    print("CARDIA RAG Pipeline - Interactive Test")
    print("=" * 60)
    print("Type 'quit' to exit.\n")
    
    # Load system instructions
    instructions_path = os.path.join(os.path.dirname(__file__), "system_instructions.txt")
    with open(instructions_path, 'r', encoding='utf-8') as f:
        system_instructions = f.read()
    
    # Initialize chat session
    chat_session = ChatSession(
        provider="openai",
        model_name="gpt-4o-mini",
        system_instruction=system_instructions
    )
    
    while True:
        try:
            user_query = input("\nYour question: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_query:
                continue
            
            print("\n" + "-" * 60)
            response = generate_response(user_query, chat_session)
            print("\n" + "=" * 60)
            print("RESPONSE:")
            print("=" * 60)
            print(response)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()


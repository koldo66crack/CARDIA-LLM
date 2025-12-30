# Streamlit UI for CARDIA Data Dictionary Assistant
# Coordinator between frontend and backend RAG pipeline

import streamlit as st
import os
from src.pipeline import generate_response
from src.conversation_manager import ChatSession


def load_system_instructions():
    """Load system instructions for the chatbot from file."""
    instructions_path = os.path.join(
        os.path.dirname(__file__), "src", "system_instructions.txt"
    )
    with open(instructions_path, 'r', encoding='utf-8') as f:
        return f.read()


def initialize_chat_session(provider: str = "openai", model_name: str = None):
    """
    Initialize a new chat session with the specified provider and model.
    
    Args:
        provider (str): LLM provider ('gemini' or 'openai'). Defaults to 'openai'.
        model_name (str): Model name for the provider. If None, uses default for that provider.
    """
    system_instructions = load_system_instructions()
    
    # Set default model names if not provided
    if model_name is None:
        if provider == "openai":
            model_name = "gpt-4o-mini"
        else:
            model_name = "gemini-2.0-flash-exp"
    
    return ChatSession(
        provider=provider,
        model_name=model_name,
        system_instruction=system_instructions
    )


# ─────────────────────────────────────────────────────────────
# STREAMLIT PAGE SETUP
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CARDIA Data Dictionary Assistant",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 CARDIA Data Dictionary Assistant")
st.markdown("Ask me about variables in the CARDIA study!")

# ─────────────────────────────────────────────────────────────
# SIDEBAR - CHAT CONTROLS
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Model Selection")
    
    # Model provider selection
    provider = st.selectbox(
        "Choose LLM Provider",
        ["OpenAI", "Gemini"],
        help="Select which LLM provider to use for generating responses"
    )
    provider_lower = provider.lower()
    
    # Set model name based on provider
    if provider_lower == "openai":
        model_name = "gpt-4o-mini"
        st.caption("📌 Model: gpt-4o-mini")
    else:
        model_name = "gemini-2.0-flash-exp"
        st.caption("📌 Model: gemini-2.0-flash-exp")
    
    st.markdown("---")
    st.markdown("## Chat Controls")
    
    if st.button("🔄 Start New Chat"):
        st.session_state.messages = []
        st.session_state.chat_session = initialize_chat_session(
            provider=provider_lower, 
            model_name=model_name
        )
        st.session_state.current_provider = provider_lower
        st.session_state.current_model = model_name
        st.rerun()  # Refresh page to show empty chat
    
    st.markdown("## RAG Settings")
    
    st.markdown("### Similarity Thresholds")
    variable_threshold = st.slider(
        "Variable Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="Minimum similarity score for retrieving relevant variables"
    )
    st.session_state.variable_threshold = variable_threshold
    
    doc_threshold = st.slider(
        "Documentation Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="Minimum similarity score for retrieving documentation"
    )
    st.session_state.doc_threshold = doc_threshold
    
    st.markdown("### Chunk Limits")
    max_variable_chunks = st.slider(
        "Max Variable Chunks",
        min_value=1,
        max_value=100,
        value=50,
        step=1,
        help="Maximum number of variables to retrieve"
    )
    st.session_state.max_variable_chunks = max_variable_chunks
    
    max_doc_chunks = st.slider(
        "Max Documentation Pages",
        min_value=1,
        max_value=30,
        value=15,
        step=1,
        help="Maximum number of documentation pages to retrieve"
    )
    st.session_state.max_doc_chunks = max_doc_chunks

# ─────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────
# Session state persists data across page reruns (when user interacts)
# Without it, chat history would disappear on every interaction

if "current_provider" not in st.session_state:
    st.session_state.current_provider = "openai"

if "current_model" not in st.session_state:
    st.session_state.current_model = "gpt-4o"

if "chat_session" not in st.session_state:
    st.session_state.chat_session = initialize_chat_session(
        provider=st.session_state.current_provider,
        model_name=st.session_state.current_model
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# ─────────────────────────────────────────────────────────────
# DISPLAY CHAT HISTORY
# ─────────────────────────────────────────────────────────────

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ─────────────────────────────────────────────────────────────
# CHAT INPUT & RESPONSE GENERATION
# ─────────────────────────────────────────────────────────────
# The walrus operator `:=` means: if user submits something, assign it to `prompt`

if prompt := st.chat_input("Ask me about CARDIA variables..."):
    
    # Display user message in chat
    with st.chat_message("user"):
        st.write(prompt)
    
    # Add user message to session state history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        # Create a status container to show progress updates
        status_container = st.empty()
        response_container = st.empty()
        
        def update_status(message: str):
            status_container.status(message, state="running")
        
        try:
            response = generate_response(
                prompt, 
                st.session_state.chat_session,
                variable_threshold=st.session_state.variable_threshold,
                doc_threshold=st.session_state.doc_threshold,
                max_variable_chunks=st.session_state.max_variable_chunks,
                max_doc_chunks=st.session_state.max_doc_chunks,
                status_callback=update_status
            )
            
            # Clear the status and show response
            status_container.empty()
            response_container.write(response)
            
            # Add assistant response to session state history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            status_container.empty()
            error_msg = f"Error generating response: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Streamlit UI for CARDIA Data Dictionary Assistant
# Coordinator between frontend and backend RAG pipeline

import streamlit as st
import os
import sys

# Add src to path so we can import modules from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llm import generate_response
from conversation_manager import ChatSession


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
    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="Minimum similarity score for retrieving relevant variables (0.0-1.0)"
    )
    st.session_state.similarity_threshold = similarity_threshold

# ─────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────
# Session state persists data across page reruns (when user interacts)
# Without it, chat history would disappear on every interaction

if "current_provider" not in st.session_state:
    st.session_state.current_provider = "gemini"

if "current_model" not in st.session_state:
    st.session_state.current_model = "gemini-2.0-flash-exp"

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
        with st.spinner("Searching CARDIA database and generating response..."):
            try:
                response = generate_response(
                    prompt, 
                    st.session_state.chat_session,
                    similarity_threshold=st.session_state.similarity_threshold
                )
                st.write(response)
                
                # Add assistant response to session state history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

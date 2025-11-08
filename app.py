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


def initialize_chat_session():
    """Initialize a new chat session with Gemini."""
    system_instructions = load_system_instructions()
    return ChatSession(
        provider="gemini",
        model_name="gemini-2.0-flash-exp",
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
    st.markdown("## Chat Controls")
    if st.button("🔄 Start New Chat"):
        st.session_state.messages = []
        st.session_state.chat_session = initialize_chat_session()
        st.rerun()  # Refresh page to show empty chat

# ─────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────
# Session state persists data across page reruns (when user interacts)
# Without it, chat history would disappear on every interaction

if "chat_session" not in st.session_state:
    st.session_state.chat_session = initialize_chat_session()

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
                response = generate_response(prompt, st.session_state.chat_session)
                st.write(response)
                
                # Add assistant response to session state history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

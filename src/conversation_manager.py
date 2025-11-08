
"""
Conversation management with provider-agnostic LLM interface.
Supports multiple LLM providers (Gemini, OpenAI, Claude) with unified conversation history.
"""

import os
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class ChatSession:
    """
    Manages conversation history and sends messages to various LLM providers.
    
    Models/clients are initialized once at setup for efficiency.
    System instructions are stored separately and handled per-provider:
    - Gemini: passed to model initialization
    - OpenAI/Claude: prepended to messages as system role
    """
    
    def __init__(self, provider: str, model_name: str, system_instruction: str):
        """
        Initialize a chat session and set up the LLM model/client.
        
        Args:
            provider (str): LLM provider ('gemini', 'openai', 'claude')
            model_name (str): Model name/ID for the provider
            system_instruction (str): System instructions for the LLM
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.history: List[Dict[str, str]] = []  # Only user/assistant exchanges
        
        # Initialize model/client based on provider
        if self.provider == "gemini":
            self.model = self._initialize_gemini()
        elif self.provider == "openai":
            self.client = self._initialize_openai()
        elif self.provider == "claude":
            self.client = self._initialize_claude()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _initialize_gemini(self):
        """Initialize and return Gemini model with system instruction."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in .env file.")
        genai.configure(api_key=api_key)
        
        return genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_instruction
        )
    
    def _initialize_openai(self):
        """Initialize and return OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")
        
        return OpenAI()
    
    def _initialize_claude(self):
        """Initialize and return Anthropic (Claude) client."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic not installed. Run: pip install anthropic")
        
        return anthropic.Anthropic()
    
    def send(self, user_msg: str, reference_context: str = None) -> str:
        """
        Send a user message and get a response from the LLM.
        Optionally includes reference material (e.g., retrieved chunks) labeled clearly for the model.
        
        Args:
            user_msg (str): User's message/query
            reference_context (str, optional): Formatted reference material from retrieval (e.g., chunks)
            
        Returns:
            str: Assistant's response
        """
        # Build augmented message with reference material if provided
        if reference_context:
            augmented_msg = f"""### REFERENCE MATERIAL (Evidence from CARDIA Data Dictionary)
{reference_context}

### USER QUERY
{user_msg}"""
        else:
            augmented_msg = user_msg
        
        # Add user message to history
        self.history.append({"role": "user", "content": augmented_msg})
        
        # Route to appropriate provider
        if self.provider == "gemini":
            response = self._send_gemini()
        elif self.provider == "openai":
            response = self._send_openai()
        elif self.provider == "claude":
            response = self._send_claude()
        
        # Add assistant response to history
        self.history.append({"role": "assistant", "content": response})
        
        return response
    
    def _send_gemini(self) -> str:
        """
        Send message to Gemini using pre-initialized model.
        
        Returns:
            str: Response text from Gemini
        """
        # Convert history to Gemini format (exclude current message for chat session)
        gemini_history = []
        for msg in self.history[:-1]:  # All except current user message
            gemini_history.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [{"text": msg["content"]}]
            })
        
        # Create chat session with history
        chat_session = self.model.start_chat(history=gemini_history)
        
        # Send the current user message (last in history)
        response = chat_session.send_message(self.history[-1]["content"])
        
        return response.text
    
    def _send_openai(self) -> str:
        """
        Send message to OpenAI using pre-initialized client.
        
        Returns:
            str: Response text from OpenAI
        """
        # Build messages with system instruction prepended
        messages = [
            {"role": "system", "content": self.system_instruction}
        ] + self.history
        
        # Send to OpenAI
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        
        return response.choices[0].message.content
    
    def _send_claude(self) -> str:
        """
        Send message to Claude using pre-initialized client.
        
        Returns:
            str: Response text from Claude
        """
        # Send to Claude with system instruction as separate parameter
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            system=self.system_instruction,
            messages=self.history
        )
        
        return response.content[0].text
    
    def clear_history(self):
        """Clear the conversation history."""
        self.history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history.
        
        Returns:
            List[Dict[str, str]]: List of {'role': ..., 'content': ...} dicts
        """
        return self.history.copy()

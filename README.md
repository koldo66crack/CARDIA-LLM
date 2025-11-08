# CARDIA RAG System

Lightweight Retrieval-Augmented Generation system for querying BIOLINCC data dictionaries. This system enables researchers to semantically search across thousands of CARDIA study variables using natural language queries, powered by AI-assisted semantic retrieval and multi-turn conversational AI.

## Disclaimer

All of the data used for this project is publicly available at https://www.cardia.dopm.uab.edu/study-information/nhlbi-data-repository-data/cardia-documentation. The information fed to the LLM consists of variable reports with metadata, not the actual entries.

## Purpose

Enable researchers to efficiently discover and understand variables in the CARDIA study by:
- Querying what variables exist across different datasets and measurement waves
- Understanding variable definitions, formats, and measurement details
- Maintaining conversation context for follow-up questions about related variables
- Getting AI-assisted explanations of variable relationships and measurement methodologies

## Overall Architecture

The CARDIA RAG system follows a classic **Retrieval-Augmented Generation** pipeline:

```
User Question
    ↓
Query Optimization (Gemini)
    ↓
Semantic Search (FAISS + embeddings)
    ↓
Relevance Filtering
    ↓
LLM Response Generation with Retrieved Context (Gemini)
    ↓
Response with Citation of Specific Variables
```

## How the Pipeline Works

### 1. **Data Preparation Phase** (One-time setup)

**`src/preprocess.py`** - CSV → JSONL Conversion
- Reads the raw BIOLINCC CSV data dictionary (`data/raw/BIOLINCC_Main Study Data Dictionary.csv`)
- Parses each variable row into a structured JSON chunk containing:
  - Variable metadata (name, dataset, data type, format, length, observation count)
  - Human-readable label/description
  - Flattened searchable content combining all fields
- Saves chunks to `data/processed/biolincc_data_dictionary.jsonl` (one JSON per line)
- Generates a preprocessing summary with dataset statistics

**`src/index.py`** - Embeddings & Vector Index Creation
- Loads all preprocessed JSONL chunks
- Initializes BGE (Beijing General Embedding) model: `BAAI/bge-small-en-v1.5`
- Generates semantic embeddings for each variable's content
  - BGE embeddings capture semantic meaning, allowing similarity matching beyond keyword search
  - Processes in batches for efficiency
- Builds a FAISS (Facebook AI Similarity Search) index
  - Uses `IndexFlatIP` with L2 normalization for cosine similarity
  - Enables ultra-fast approximate nearest-neighbor search
- Saves three files to `data/processed/`:
  - `faiss_index.bin` - Binary FAISS index
  - `metadata.pkl` - Python pickle of all variable metadata
  - `index_info.json` - Index configuration and statistics

### 2. **Query Processing Phase** (Per user question)

**`src/llm.py`** - RAG Orchestration
The core query pipeline that ties together retrieval and generation:

1. **Query Optimization**: 
   - User's natural language question is sent to Gemini 2.0 Flash
   - LLM rewrites it as an optimized search query, incorporating conversation history for context
   - Example: "Do you have any blood pressure stuff?" → "blood pressure measurements and hypertension variables"

2. **Semantic Retrieval**:
   - Optimized query is embedded using the BGE model (same model as preprocessing)
   - Performs similarity search against FAISS index for k=10 nearest neighbors
   - Returns scored results based on cosine similarity

3. **Relevance Filtering**:
   - Filters results by similarity threshold (default: 0.75)
   - Only includes high-confidence matches to avoid hallucination
   - If no results exceed threshold, proceeds without retrieved context

4. **Context Building**:
   - If relevant results found, formats them into a structured context string
   - Includes variable name, dataset, description, type, format, observation count
   - Adds relevance scores for transparency

5. **LLM Response Generation**:
   - Sends user query + retrieved context to Gemini via ChatSession
   - LLM generates response grounded in the specific variables found
   - System instructions guide the LLM to cite variable names and explain differences

**`src/conversation_manager.py`** - Multi-turn Conversation Management
- Manages conversation history across turns
- Supports multiple LLM providers (Gemini, OpenAI, Claude)
- Handles provider-specific formatting requirements:
  - **Gemini**: System instruction passed to model initialization
  - **OpenAI/Claude**: System instruction prepended as system role message
- Maintains full conversation history for context awareness
- Reference material (retrieved chunks) is clearly labeled in augmented user messages

**`src/rag_retriever.py`** - Retrieval & Context Formatting
- **`search_variables()`**: Main search function
  - Loads existing FAISS index and metadata from disk
  - Embeds user query and searches index
  - Returns scored chunks
  
- **`build_retrieved_context()`**: Formats chunks for LLM
  - Creates human-readable context string with clear section headers
  - Includes all relevant metadata (dataset, type, format, observations)
  - Shows relevance scores for transparency

### 3. **Frontend & User Interaction** (`app.py`)

**Streamlit Web Interface**
- Provides chat-based UI for querying the system
- Session state management maintains:
  - Conversation history across interactions
  - Chat session object for multi-turn context
- User can start new conversations or continue existing ones
- Real-time streaming of responses with loading indicators
- Error handling and graceful degradation

## Directory Structure

```
CARDIA_LLM/
├── app.py                                    # Streamlit UI entry point
├── requirements.txt                          # Python dependencies
├── README.md                                 # This file
├── data/
│   ├── raw/                                  # Original source files
│   │   └── BIOLINCC_Main Study Data Dictionary.csv
│   └── processed/                            # Generated artifacts
│       ├── biolincc_data_dictionary.jsonl   # Preprocessed chunks
│       ├── faiss_index.bin                  # Vector search index
│       ├── metadata.pkl                     # Chunk metadata
│       ├── index_info.json                  # Index configuration
│       └── preprocessing_summary.json       # Preprocessing statistics
└── src/
    ├── app.py                               # (duplicate for reference)
    ├── preprocess.py                        # CSV → JSONL conversion
    ├── index.py                             # Embedding & index building
    ├── llm.py                               # RAG pipeline orchestration
    ├── conversation_manager.py              # Multi-turn conversation management
    ├── rag_retriever.py                     # Semantic search & retrieval
    └── system_instructions.txt              # LLM system prompt
```

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
The system requires a Google Generative AI (Gemini) API key:

1. **Get your API key** from [Google AI Studio](https://makersuite.google.com/app/apikey)

2. **Create a `.env` file** (copy from `.env.example`):
   ```bash
   cp .env.example .env
   ```

3. **Add your API key** to `.env`:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

⚠️ **Important**: The `.env` file is automatically ignored by git (see `.gitignore`). Never commit this file to version control!

Optional API keys for alternative models:
- **OpenAI**: Set `OPENAI_API_KEY` to use GPT-4 or GPT-3.5
- **Anthropic Claude**: Set `ANTHROPIC_API_KEY` to use Claude models

### 3. Prepare Data (First Time Only)
```bash
# Convert CSV to JSONL chunks
python src/preprocess.py

# Build FAISS index and embeddings
python src/index.py
```

This creates the processed data files needed for semantic search.

### 4. Run the Interactive UI
```bash
streamlit run app.py
```

This opens the chatbot at `http://localhost:8501`

### 5. Alternative: Command-Line Interface
```bash
python src/llm.py
```

For non-interactive testing and debugging.

## How to Use the Chatbot

1. **Ask natural language questions** about CARDIA variables:
   - "What blood pressure variables are available?"
   - "How is diabetes measured in the study?"
   - "Which datasets have cholesterol measurements?"

2. **Follow-up questions** automatically maintain context:
   - Previous conversation is remembered for better interpretation
   - You can ask clarifying questions about variables mentioned earlier

3. **Review retrieved variables** in the response:
   - The assistant cites specific variable names and datasets
   - Explanations include measurement formats and data types
   - Relevance scores show confidence in the match

4. **Start a new conversation** with the "Start New Chat" button in the sidebar

## Key Components Explained

### Embedding Model: BGE (BAAI/bge-small-en-v1.5)
- State-of-the-art general embedding model
- Small footprint (~20MB) but effective for domain-specific search
- Captures semantic meaning better than keyword matching
- Pre-trained on 215M text pairs from various domains

### Vector Search: FAISS IndexFlatIP
- **IndexFlatIP**: Inner product index (equivalent to cosine similarity on normalized vectors)
- Stores all embedding vectors in-memory for exact search
- Search time: O(n) but extremely fast in practice (<100ms for 10k vectors)
- Cosine similarity metric: measures angle between embedding vectors (0.0 = different, 1.0 = identical)

### LLM: Gemini 2.0 Flash
- Fast, efficient model for query optimization and response generation
- Supports system instructions for role-based behavior
- Multi-turn conversation support with full history
- Providers: Can swap to OpenAI (GPT-4, GPT-3.5) or Claude (Claude 3) with minimal code changes

### Similarity Threshold
- Default: 0.75 (75% match confidence)
- If no results exceed threshold, generates response without retrieved context
- Prevents low-confidence hallucinations while allowing fallback responses
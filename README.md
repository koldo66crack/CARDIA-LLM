# CARDIA RAG System

# [View the live app here](https://cardia-llm.streamlit.app/)

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

The CARDIA RAG system uses a **Two-Stage Retrieval-Augmented Generation** pipeline:

```
User Question
    ↓
Stage 1: Variable Search
    ├─ Query Optimization (GPT-4o-mini)
    └─ Hybrid Search: Keyword + Semantic (FAISS + BGE embeddings)
    ↓
Sufficiency Check (GPT-4o-mini)
    ├─ Is variable info sufficient? YES → Generate Response
    └─ Is variable info sufficient? NO ↓
    ↓
Stage 2: Documentation Search
    ├─ Enhanced query (includes dataset names from Stage 1)
    └─ Hybrid Search: Keyword + Semantic (FAISS + BGE embeddings)
    ↓
Response Generation (Gemini 2.0 Flash / GPT-4o)
    ↓
Response with Citations from Variables + Documentation
```

## How the Pipeline Works

### 1. **Data Preparation Phase** (One-time setup)

**`preprocessing/csv_preprocessor.py`** - CSV → JSONL Conversion
- Reads the raw BIOLINCC CSV data dictionary (`data/raw/BIOLINCC_Main Study Data Dictionary.csv`)
- Parses each variable row into a structured JSON chunk containing:
  - Variable metadata (name, dataset, data type, format, length, observation count)
  - Human-readable label/description
  - Flattened searchable content combining all fields
- Saves chunks to `data/processed/biolincc_data_dictionary.jsonl` (one JSON per line)
- Generates a preprocessing summary with dataset statistics

**`extract_docs.py`** - PDF/DOC → JSONL Extraction
- Extracts text from CARDIA study documentation (PDFs and Word docs)
- Uses multi-tier approach: pdfplumber → Tesseract OCR → Gemini Vision (VLM)
- Enriches each page chunk with metadata: filename, wave, year, document type, page number
- Saves chunks to `data/processed/cardia_documentation.jsonl`

**`src/index.py`** - Embeddings & Vector Index Creation
- Builds **two separate FAISS indices**:
  1. **Variable Index** (`faiss_index.bin`, `metadata.pkl`, `index_info.json`)
  2. **Documentation Index** (`docs_faiss_index.bin`, `docs_metadata.pkl`, `docs_index_info.json`)
- Uses BGE (Beijing General Embedding) model: `BAAI/bge-small-en-v1.5`
- For documentation chunks, enriches content with metadata (filename, year, wave) before embedding
- Uses `IndexFlatIP` with L2 normalization for cosine similarity search

### 2. **Query Processing Phase** (Per user question)

**`src/pipeline.py`** - Two-Stage RAG Orchestration
The main entry point that coordinates the entire pipeline:

**Stage 1: Variable Retrieval**
1. **Query Optimization** (`src/query_optimizer.py`):
   - User question sent to GPT-4o-mini
   - LLM rewrites it as an optimized search query, incorporating conversation history
   - Extracts keywords/tags for hybrid search
   - Example: "Do you have any blood pressure stuff?" → query: "blood pressure measurements and hypertension variables", tags: ["blood pressure", "hypertension"]

2. **Hybrid Variable Search** (`src/retrieval/variables.py`):
   - **Keyword Search**: Exact/substring matches on variable name, dataset, label fields
   - **Semantic Search**: Embeds query with BGE, searches FAISS index for similar variables
   - Merges results (keyword matches prioritized, then semantic matches)
   - Filters by similarity threshold (default: 0.75)

**Stage 2: Sufficiency Check**
3. **LLM Decision** (`src/query_optimizer.py`):
   - Sends original question + retrieved variables to GPT-4o-mini
   - Uses chain-of-thought prompting to evaluate sufficiency
   - Returns:
     - **sufficient**: Boolean (can we answer with just variables?)
     - **reason**: Explanation of decision
     - **doc_query**: Optimized query for documentation search (if needed)
     - **datasets**: Dataset names to use as keyword filters

**Stage 3: Documentation Search (if insufficient)**
4. **Hybrid Documentation Search** (`src/retrieval/docs.py`):
   - **Keyword Search**: Matches dataset names against document filenames
   - **Semantic Search**: Embeds doc_query with BGE, searches documentation FAISS index
   - Merges and filters results (default threshold: 0.50)

**Stage 4: Response Generation**
5. **Context Building** (`src/context_builder.py`):
   - Formats variable chunks with metadata (name, dataset, description, type, observations)
   - Formats doc chunks with metadata (filename, wave, year, page, content snippet)
   - Combines into structured context string

6. **LLM Response** (`src/conversation_manager.py`):
   - Sends question + context to generation LLM (Gemini 2.0 Flash or GPT-4o)
   - LLM generates response grounded in retrieved information
   - System instructions guide proper citation and explanation

**`src/retrieval/`** - Modular Retrieval Components
- **`search.py`**: Generic search utilities (load_index, semantic_search, keyword_search, merge_results)
- **`variables.py`**: Variable-specific search and metadata extraction
- **`docs.py`**: Documentation-specific search and source grouping

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
├── extract_docs.py                           # Documentation extraction runner
├── setup_variables.py                        # Variables setup runner
├── requirements.txt                          # Python dependencies
├── README.md                                 # This file
├── data/
│   ├── raw/                                  # Original source files (gitignored)
│   │   ├── BIOLINCC_Main Study Data Dictionary.csv
│   │   └── CARDIA documentation/            # PDFs and Word docs
│   └── processed/                            # Generated artifacts
│       ├── biolincc_data_dictionary.jsonl   # Variable chunks
│       ├── cardia_documentation.jsonl       # Documentation chunks
│       ├── faiss_index.bin                  # Variable vector index
│       ├── metadata.pkl                     # Variable metadata
│       ├── index_info.json                  # Variable index config
│       ├── docs_faiss_index.bin             # Documentation vector index
│       ├── docs_metadata.pkl                # Documentation metadata
│       ├── docs_index_info.json             # Documentation index config
│       └── preprocessing_summary.json       # Processing statistics
├── preprocessing/
│   ├── csv_preprocessor.py                  # CSV → JSONL conversion
│   ├── pdf_extractor.py                     # PDF text extraction
│   ├── doc_extractor.py                     # Word doc extraction
│   ├── vlm_utils.py                         # Gemini Vision for OCR fallback
│   └── ocr_quality.py                       # OCR quality assessment
└── src/
    ├── pipeline.py                          # Main RAG orchestrator
    ├── query_optimizer.py                   # Query optimization & sufficiency check
    ├── context_builder.py                   # Context formatting for LLM
    ├── conversation_manager.py              # Multi-turn conversation management
    ├── index.py                             # Embedding & index building
    ├── retrieval/
    │   ├── search.py                        # Generic search utilities
    │   ├── variables.py                     # Variable-specific retrieval
    │   └── docs.py                          # Documentation-specific retrieval
    ├── prompts/
    │   └── sufficiency_check.txt            # Sufficiency check prompt
    ├── rag_query_optimization_prompt.txt    # Query optimization prompt
    └── system_instructions.txt              # LLM system prompt for generation
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

**Step 1: Setup Variables**
```bash
python setup_variables.py
```
This will:
- Convert the BIOLINCC CSV to JSONL chunks
- Build the variable FAISS index and embeddings

**Step 2: Extract Documentation** (Optional but recommended)
```bash
python extract_docs.py
```
This will:
- Extract text from PDFs and Word documents in `data/raw/CARDIA documentation/`
- Use multi-tier extraction: pdfplumber → Tesseract OCR → Gemini Vision
- Save documentation chunks to JSONL

**Step 3: Build Documentation Index** (Required if you ran Step 2)
```bash
python src/index.py --build docs
```
This creates the documentation FAISS index for the second retrieval stage.

### 4. Run the Interactive UI
```bash
streamlit run app.py
```

This opens the chatbot at `http://localhost:8501`

### 5. Alternative: Command-Line Interface
```bash
python src/pipeline.py
```

For non-interactive testing and debugging of the RAG pipeline.

## How to Use the Chatbot

1. **Ask natural language questions** about CARDIA variables and study procedures:
   - "What blood pressure variables are available?"
   - "How is diabetes measured in the study?"
   - "Which datasets have cholesterol measurements?"
   - "What procedures were followed during the second wave exams?"

2. **Two-stage retrieval automatically adapts**:
   - System first searches for relevant variables
   - If variable info is sufficient, generates response immediately
   - If more context needed, searches documentation for detailed procedures
   - You'll see progress updates showing which stage is active

3. **Follow-up questions** automatically maintain context:
   - Previous conversation is remembered for better interpretation
   - You can ask clarifying questions about variables or procedures mentioned earlier

4. **Adjust RAG settings** in the sidebar:
   - **Similarity Thresholds**: Control how strict the relevance filtering is
   - **Chunk Limits**: Cap the number of variables/documentation pages retrieved
   - Lower chunk limits if you hit token limits with complex queries

5. **Start a new conversation** with the "🔄 Start New Chat" button in the sidebar

## Key Components Explained

### Embedding Model: BGE (BAAI/bge-small-en-v1.5)
- State-of-the-art general embedding model
- Small footprint (~20MB) but effective for domain-specific search
- Captures semantic meaning better than keyword matching alone
- Pre-trained on 215M text pairs from various domains
- Used for both variable and documentation indices

### Vector Search: FAISS IndexFlatIP
- **IndexFlatIP**: Inner product index (equivalent to cosine similarity on normalized vectors)
- **Two separate indices**: One for variables, one for documentation
- Stores all embedding vectors in-memory for exact search
- Search time: O(n) but extremely fast in practice (<100ms for 10k vectors)
- Cosine similarity metric: measures angle between embedding vectors (0.0 = different, 1.0 = identical)

### Hybrid Search (Keyword + Semantic)
- **Keyword Search**: Fast exact/substring matching on metadata fields
- **Semantic Search**: Finds conceptually similar content via embeddings
- Results merged with keyword matches prioritized (higher precision)
- Combines recall of semantic search with precision of keyword matching

### LLMs: Multi-Model Strategy
- **Query Optimization & Sufficiency Check**: GPT-4o-mini (fast, cost-effective for JSON-structured tasks)
- **Response Generation**: Gemini 2.0 Flash or GPT-4o (configurable, optimized for natural language)
- Supports OpenAI, Gemini, and Claude providers via conversation_manager
- Chain-of-thought prompting for sufficiency check improves decision quality

### Retrieval Parameters
- **Variable Threshold**: Default 0.75 (strict filtering for variable matches)
- **Documentation Threshold**: Default 0.50 (more lenient for contextual info)
- **Max Chunks**: Default 50 variables, 15 doc pages (prevents token overflow)
- Adjustable via Streamlit sidebar sliders
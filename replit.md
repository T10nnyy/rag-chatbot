# RAG-Based Website Chatbot

## Overview

A RAG (Retrieval-Augmented Generation) chatbot application that allows users to crawl any website, build a semantic knowledge base from its content, and ask natural language questions. The system crawls websites up to configurable depth, processes content into chunks, generates vector embeddings, and uses semantic search to provide accurate answers based on the indexed content.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend
- **Streamlit** serves as the web UI framework
- Single-page application with URL input, crawl controls, and chat interface
- Session state management for persistence across interactions

### Backend Pipeline
The application follows a modular RAG architecture with four core components:

1. **Web Crawler (`crawler.py`)**
   - Fetches and parses HTML pages using requests and BeautifulSoup
   - Configurable crawl depth and page limits
   - URL validation and normalization
   - Filters out non-HTML content (PDFs, images, scripts, etc.)

2. **Text Processor (`text_processor.py`)**
   - Cleans and normalizes extracted text
   - Token-based chunking using tiktoken (cl100k_base encoding)
   - Creates overlapping chunks (500 tokens default, 100 token overlap)
   - Preserves source metadata (URL, title) with each chunk

3. **Vector Store (`vector_store.py`)**
   - Generates embeddings using OpenAI's text-embedding-3-small model
   - Stores vectors in FAISS index for efficient similarity search
   - In-memory storage (no persistent database)

4. **RAG Chatbot (`rag_chatbot.py`)**
   - Retrieves relevant chunks via semantic similarity search
   - Formats context from multiple sources
   - Generates responses using OpenAI GPT model
   - Maintains conversation history

### Design Decisions

**In-Memory Vector Storage with FAISS**
- Problem: Need fast semantic search over website content
- Solution: FAISS (Facebook AI Similarity Search) for vector indexing
- Rationale: No persistent storage needed since content is re-crawled per session; FAISS provides fast similarity search without database overhead

**Token-Based Chunking with Overlap**
- Problem: LLMs have context limits; need to split content intelligently
- Solution: 500-token chunks with 100-token overlap
- Rationale: Overlapping ensures context isn't lost at chunk boundaries; token-based (vs character-based) aligns with LLM processing

**Modular Component Design**
- Problem: Complex RAG pipeline with multiple stages
- Solution: Separate modules for crawling, processing, storage, and chat
- Rationale: Easy to test, modify, or swap individual components

## External Dependencies

### APIs
- **OpenAI API** (required)
  - Embeddings: `text-embedding-3-small` for vector generation
  - Chat: `gpt-5` for response generation
  - Requires `OPENAI_API_KEY` environment variable

### Python Libraries
- `streamlit` - Web UI framework
- `openai` - OpenAI API client
- `requests` - HTTP requests for crawling
- `beautifulsoup4` - HTML parsing
- `validators` - URL validation
- `tiktoken` - Token counting for chunking
- `faiss-cpu` - Vector similarity search
- `numpy` - Numerical operations for embeddings

### No Database Required
- All data is stored in-memory during session
- Vector index and chunks are rebuilt on each crawl
- No persistent storage layer needed
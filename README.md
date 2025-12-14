# RAG-Based Website Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that can answer questions about any website by crawling its content and building a semantic knowledge base.

## Overview

This application allows users to:
1. Input any website URL
2. Automatically crawl the website (up to 2 levels deep)
3. Build a searchable knowledge base using vector embeddings
4. Ask natural language questions and get accurate answers based on the website content

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface (Streamlit)                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │   URL Input     │  │  Crawl Button   │  │   Chat Interface    │  │
│  └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘  │
└───────────┼─────────────────────┼──────────────────────┼────────────┘
            │                     │                      │
            ▼                     ▼                      ▼
┌───────────────────────────────────────────────────────────────────┐
│                        Application Layer                           │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     Web Crawler Module                       │   │
│  │  • Fetches HTML pages from base URL                         │   │
│  │  • Extracts titles, headings, visible text                  │   │
│  │  • Follows links up to 2 levels deep                        │   │
│  │  • Filters out scripts, images, ads, non-HTML               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   Text Processor Module                      │   │
│  │  • Cleans and normalizes extracted text                     │   │
│  │  • Splits content into overlapping chunks (500 tokens)      │   │
│  │  • Preserves source metadata (URL, title)                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Vector Store Module                       │   │
│  │  • Generates embeddings (OpenAI text-embedding-3-small)     │   │
│  │  • Stores vectors in FAISS index                            │   │
│  │  • Performs semantic similarity search                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    RAG Chatbot Module                        │   │
│  │  • Retrieves top-k relevant chunks for user query           │   │
│  │  • Constructs prompt with retrieved context                 │   │
│  │  • Generates answer using GPT-5                             │   │
│  │  • Provides source citations                                │   │
│  └─────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│                        External Services                           │
│  ┌────────────────────────┐    ┌────────────────────────────────┐ │
│  │      OpenAI API        │    │         FAISS Index            │ │
│  │  • Embeddings          │    │  • In-memory vector storage    │ │
│  │  • Chat Completions    │    │  • Fast similarity search      │ │
│  └────────────────────────┘    └────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

## Features

### URL Input & Crawling
- Accepts any valid HTTP/HTTPS URL
- Configurable crawl depth (1-2 levels)
- Configurable maximum pages (10-100)
- Respects same-domain policy
- Filters non-HTML content automatically

### Content Extraction
- Extracts page titles and headings
- Captures visible text content
- Removes scripts, styles, ads, and navigation
- Cleans and normalizes extracted text

### Knowledge Base
- Token-aware text chunking (500 tokens with 100 token overlap)
- OpenAI embeddings (text-embedding-3-small)
- FAISS vector index for fast similarity search
- Source metadata preservation for citations

### Chat Interface
- Natural language question input
- Context-aware answer generation using GPT-5
- Source citations with links
- Conversation history support
- Clear chat/reset functionality

### Error Handling
- Invalid URL validation
- Timeout handling for slow websites
- Graceful handling of blocked/inaccessible sites
- HTTP error code handling
- Content parsing error recovery

## Example Queries

After crawling a website, you can ask questions like:

- "What is this website about?"
- "What services do they offer?"
- "What are the main products listed?"
- "Can you summarize the pricing information?"
- "What contact information is available?"
- "What is the company's mission statement?"

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Web Crawling | BeautifulSoup4, Requests |
| Text Processing | tiktoken |
| Vector Database | FAISS |
| Embeddings | OpenAI text-embedding-3-small |
| LLM | OpenAI GPT-5 |
| Language | Python 3.11 |

## Project Structure

```
├── app.py              # Streamlit UI application
├── crawler.py          # Web crawler module
├── text_processor.py   # Text chunking and preprocessing
├── vector_store.py     # FAISS vector store and embeddings
├── rag_chatbot.py      # RAG pipeline and chat generation
├── README.md           # This file
└── .streamlit/
    └── config.toml     # Streamlit configuration
```

## Setup & Installation

1. **Clone the repository**

2. **Set up environment variables**
   - Add your OpenAI API key to the Secrets tab:
   - `OPENAI_API_KEY`: Your OpenAI API key

3. **Run the application**
   ```bash
   streamlit run app.py --server.port 5000
   ```

## Usage

1. Open the application in your browser
2. Enter a website URL in the sidebar
3. Configure crawl depth and maximum pages
4. Click "Crawl & Build Knowledge Base"
5. Wait for crawling and indexing to complete
6. Ask questions about the website content

## Limitations

1. **Rate Limiting**: The crawler includes 0.3s delays between requests to avoid overwhelming servers
2. **JavaScript Content**: Cannot crawl JavaScript-rendered content (SPA applications)
3. **Authentication**: Cannot access pages behind login/authentication
4. **Robots.txt**: Does not currently respect robots.txt (use responsibly)
5. **Session Storage**: Knowledge base is stored in memory and lost on restart
6. **File Size**: Very large websites may hit the maximum page limit
7. **Content Types**: Only processes HTML pages, ignores PDFs and other formats

## Future Enhancements

1. **Persistent Storage**: Save and load knowledge bases to/from disk
2. **Multiple URLs**: Support crawling multiple websites
3. **PDF Support**: Extract and index PDF documents
4. **Advanced Chunking**: Semantic chunking based on document structure
5. **Caching**: Cache crawled content to avoid re-crawling
6. **Robots.txt**: Respect website crawling preferences
7. **Sitemap Support**: Use sitemaps for more efficient crawling
8. **Export/Import**: Export knowledge base for sharing

## API Requirements

This application requires an OpenAI API key with access to:
- `text-embedding-3-small` (embeddings)
- `gpt-5` (chat completions)

## License

MIT License

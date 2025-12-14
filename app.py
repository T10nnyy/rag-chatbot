"""
RAG-Based Website Chatbot - Streamlit UI
A chatbot that can answer questions about any website by crawling and indexing its content.
"""

import streamlit as st
import os
from crawler import WebCrawler
from text_processor import TextProcessor
from vector_store import VectorStore
from rag_chatbot import RAGChatbot

st.set_page_config(
    page_title="Website RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStore()
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot(st.session_state.vector_store)
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'crawled_url' not in st.session_state:
        st.session_state.crawled_url = None
    if 'crawl_stats' not in st.session_state:
        st.session_state.crawl_stats = None
    if 'is_indexed' not in st.session_state:
        st.session_state.is_indexed = False

def check_api_key():
    """Check if OpenAI API key is configured"""
    return os.environ.get("OPENAI_API_KEY") is not None

def crawl_and_index_website(url: str, max_depth: int, max_pages: int):
    """Crawl website and build knowledge base"""
    progress_container = st.empty()
    status_container = st.empty()
    
    with progress_container.container():
        st.write("**Phase 1: Crawling Website**")
        crawl_progress = st.progress(0)
        crawl_status = st.empty()
    
    crawler = WebCrawler(max_depth=max_depth, max_pages=max_pages)
    
    def crawl_callback(count, current_url, title):
        progress = min(count / max_pages, 1.0)
        crawl_progress.progress(progress)
        crawl_status.text(f"Crawled {count} pages | Current: {title[:50]}...")
    
    try:
        crawled_data = crawler.crawl(url, progress_callback=crawl_callback)
        crawl_summary = crawler.get_crawl_summary()
    except ValueError as e:
        st.error(f"Invalid URL: {e}")
        return False
    except Exception as e:
        st.error(f"Crawling error: {e}")
        return False
    
    if not crawled_data:
        st.warning("No content could be extracted from the website. The site may block crawling or have no accessible content.")
        return False
    
    progress_container.empty()
    
    with status_container.container():
        st.success(f"Crawled {crawl_summary['pages_crawled']} pages successfully!")
        st.write("**Phase 2: Processing Text**")
        process_progress = st.progress(0)
    
    processor = TextProcessor(chunk_size=500, chunk_overlap=100)
    chunks = processor.process_crawled_data(crawled_data)
    process_stats = processor.get_processing_stats(chunks)
    
    process_progress.progress(1.0)
    
    if not chunks:
        st.warning("No text chunks could be created from the crawled content.")
        return False
    
    status_container.empty()
    
    with st.container():
        st.write("**Phase 3: Building Knowledge Base**")
        embed_progress = st.progress(0)
        embed_status = st.empty()
    
    def embed_callback(current, total):
        embed_progress.progress(current / total)
        embed_status.text(f"Embedding chunks: {current}/{total}")
    
    try:
        st.session_state.vector_store.build_index(chunks, progress_callback=embed_callback)
    except Exception as e:
        st.error(f"Error building knowledge base: {e}")
        return False
    
    st.session_state.crawled_url = url
    st.session_state.crawl_stats = {
        **crawl_summary,
        **process_stats
    }
    st.session_state.is_indexed = True
    st.session_state.messages = []
    st.session_state.chatbot.clear_history()
    
    return True

def display_chat_interface():
    """Display the chat interface"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.markdown(f"- [{source['title']}]({source['url']}) (relevance: {source['relevance_score']})")
    
    if prompt := st.chat_input("Ask a question about the website..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.generate_answer(prompt)
            
            st.markdown(response['answer'])
            
            if response.get('sources'):
                with st.expander("📚 Sources"):
                    for source in response['sources']:
                        st.markdown(f"- [{source['title']}]({source['url']}) (relevance: {source['relevance_score']})")
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response['answer'],
            "sources": response.get('sources', [])
        })

def main():
    initialize_session_state()
    
    st.title("🤖 Website RAG Chatbot")
    st.markdown("*Crawl any website and ask questions about its content*")
    
    if not check_api_key():
        st.error("⚠️ OpenAI API Key is not configured. Please add your OPENAI_API_KEY to the Secrets.")
        st.info("Go to the Secrets tab in Replit and add your OpenAI API key.")
        return
    
    with st.sidebar:
        st.header("🌐 Website Configuration")
        
        url = st.text_input(
            "Website URL",
            placeholder="https://example.com",
            help="Enter the URL of the website you want to crawl"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            max_depth = st.selectbox(
                "Crawl Depth",
                options=[1, 2],
                index=1,
                help="How deep to crawl (1 = homepage only, 2 = homepage + linked pages)"
            )
        with col2:
            max_pages = st.selectbox(
                "Max Pages",
                options=[10, 25, 50, 100],
                index=1,
                help="Maximum number of pages to crawl"
            )
        
        if st.button("🚀 Crawl & Build Knowledge Base", type="primary", use_container_width=True):
            if not url:
                st.error("Please enter a URL")
            else:
                with st.spinner("Processing..."):
                    success = crawl_and_index_website(url, max_depth, max_pages)
                    if success:
                        st.success("Knowledge base built successfully!")
                        st.rerun()
        
        st.divider()
        
        if st.session_state.crawl_stats:
            st.header("📊 Knowledge Base Stats")
            stats = st.session_state.crawl_stats
            
            st.metric("Pages Crawled", stats.get('pages_crawled', 0))
            st.metric("Text Chunks", stats.get('total_chunks', 0))
            st.metric("Total Tokens", f"{stats.get('total_tokens', 0):,}")
            
            if st.session_state.crawled_url:
                st.caption(f"Source: {st.session_state.crawled_url}")
        
        st.divider()
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chatbot.clear_history()
            st.rerun()
        
        if st.button("🔄 Reset Everything", use_container_width=True):
            st.session_state.vector_store.clear()
            st.session_state.messages = []
            st.session_state.chatbot.clear_history()
            st.session_state.crawled_url = None
            st.session_state.crawl_stats = None
            st.session_state.is_indexed = False
            st.rerun()
    
    if st.session_state.is_indexed:
        st.success(f"✅ Knowledge base ready! Ask questions about: **{st.session_state.crawled_url}**")
        display_chat_interface()
    else:
        st.info("👈 Enter a website URL in the sidebar and click 'Crawl & Build Knowledge Base' to get started.")
        
        st.markdown("---")
        st.markdown("### How it works")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**1. Enter URL**")
            st.markdown("Provide a website URL to crawl")
        
        with col2:
            st.markdown("**2. Crawl Website**")
            st.markdown("System extracts text from pages")
        
        with col3:
            st.markdown("**3. Build Index**")
            st.markdown("Content is chunked and embedded")
        
        with col4:
            st.markdown("**4. Ask Questions**")
            st.markdown("Chat with your knowledge base")

if __name__ == "__main__":
    main()

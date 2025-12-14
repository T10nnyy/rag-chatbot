"""
RAG-Based Website Chatbot - Streamlit UI
A chatbot that can answer questions about any website by crawling and indexing its content.
Supports both OpenAI and Google Gemini models.
"""

import streamlit as st
import os
import time
from crawler import WebCrawler
from text_processor import TextProcessor
from vector_store import VectorStore
from rag_chatbot import RAGChatbot
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Website RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'embedding_provider' not in st.session_state:
        st.session_state.embedding_provider = "openai"
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStore(
            embedding_provider=st.session_state.embedding_provider
        )
    if 'selected_provider' not in st.session_state:
        st.session_state.selected_provider = "gemini"
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "gemini-2.0-flash"
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot(
            st.session_state.vector_store,
            model=st.session_state.selected_model,
            provider=st.session_state.selected_provider
        )
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'crawled_urls' not in st.session_state:
        st.session_state.crawled_urls = []
    if 'crawl_stats' not in st.session_state:
        st.session_state.crawl_stats = None
    if 'is_indexed' not in st.session_state:
        st.session_state.is_indexed = False
    if 'all_crawl_stats' not in st.session_state:
        st.session_state.all_crawl_stats = []

def check_api_keys():
    """Check which API keys are configured"""
    return {
        'openai': os.environ.get("OPENAI_API_KEY") is not None,
        'gemini': os.environ.get("GEMINI_API_KEY") is not None
    }

def update_chatbot_model(provider: str, model: str):
    """Update the chatbot with a new model"""
    st.session_state.selected_provider = provider
    st.session_state.selected_model = model
    st.session_state.chatbot = RAGChatbot(
        st.session_state.vector_store,
        model=model,
        provider=provider
    )
    # Preserve conversation history
    if hasattr(st.session_state, 'messages'):
        for msg in st.session_state.messages:
            if msg['role'] == 'user':
                st.session_state.chatbot.conversation_history.append({
                    'role': 'user',
                    'content': msg['content']
                })
            elif msg['role'] == 'assistant':
                st.session_state.chatbot.conversation_history.append({
                    'role': 'assistant',
                    'content': msg['content']
                })

def update_embedding_provider(provider: str):
    """Update the embedding provider and recreate vector store"""
    if provider != st.session_state.embedding_provider:
        st.session_state.embedding_provider = provider
        st.session_state.vector_store = VectorStore(embedding_provider=provider)
        st.session_state.chatbot = RAGChatbot(
            st.session_state.vector_store,
            model=st.session_state.selected_model,
            provider=st.session_state.selected_provider
        )
        # Clear index as embeddings are incompatible
        st.session_state.is_indexed = False
        st.session_state.crawled_urls = []
        st.session_state.crawl_stats = None
        st.session_state.all_crawl_stats = []
        st.session_state.messages = []
        st.warning("⚠️ Embedding provider changed. Please re-crawl websites to rebuild the knowledge base.")

def crawl_and_index_website(url: str, max_depth: int, max_pages: int, append_mode: bool = False):
    """Crawl website and build knowledge base"""
    start_time = time.time()
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
    
    crawl_time = time.time() - start_time
    
    if not crawled_data:
        st.warning("No content could be extracted from the website. The site may block crawling or have no accessible content.")
        return False
    
    progress_container.empty()
    
    with status_container.container():
        st.success(f"Crawled {crawl_summary['pages_crawled']} pages in {crawl_time:.1f}s!")
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
    
    embed_start_time = time.time()
    with st.container():
        st.write("**Phase 3: Building Knowledge Base**")
        embed_progress = st.progress(0)
        embed_status = st.empty()
    
    def embed_callback(current, total):
        embed_progress.progress(current / total)
        embed_status.text(f"Embedding chunks: {current}/{total}")
    
    try:
        if append_mode and st.session_state.is_indexed:
            st.session_state.vector_store.add_chunks(chunks, progress_callback=embed_callback)
        else:
            st.session_state.vector_store.build_index(chunks, progress_callback=embed_callback)
    except Exception as e:
        st.error(f"Error building knowledge base: {e}")
        return False
    
    embed_time = time.time() - embed_start_time
    total_time = time.time() - start_time
    
    this_crawl_stats = {
        'url': url,
        'pages_crawled': crawl_summary['pages_crawled'],
        'total_chunks': process_stats['total_chunks'],
        'total_tokens': process_stats['total_tokens'],
        'crawl_time': round(crawl_time, 1),
        'embed_time': round(embed_time, 1),
        'total_time': round(total_time, 1),
        'pages': crawl_summary.get('pages', [])
    }
    
    if not append_mode:
        st.session_state.all_crawl_stats = []
        st.session_state.crawled_urls = []
    
    st.session_state.all_crawl_stats.append(this_crawl_stats)
    
    if url not in st.session_state.crawled_urls:
        st.session_state.crawled_urls.append(url)
    
    total_stats = {
        'pages_crawled': sum(s['pages_crawled'] for s in st.session_state.all_crawl_stats),
        'total_chunks': sum(s['total_chunks'] for s in st.session_state.all_crawl_stats),
        'total_tokens': sum(s['total_tokens'] for s in st.session_state.all_crawl_stats),
        'total_time': sum(s['total_time'] for s in st.session_state.all_crawl_stats),
        'urls_count': len(st.session_state.crawled_urls)
    }
    st.session_state.crawl_stats = total_stats
    
    st.session_state.is_indexed = True
    
    if not append_mode:
        st.session_state.messages = []
        st.session_state.chatbot.clear_history()
    
    return True

def display_crawl_statistics():
    """Display detailed crawl statistics dashboard"""
    if not st.session_state.all_crawl_stats:
        return
    
    st.subheader("📊 Crawl Statistics Dashboard")
    
    stats = st.session_state.crawl_stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Pages", stats.get('pages_crawled', 0))
    with col2:
        st.metric("Text Chunks", stats.get('total_chunks', 0))
    with col3:
        st.metric("Total Tokens", f"{stats.get('total_tokens', 0):,}")
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("URLs Indexed", stats.get('urls_count', 0))
    with col5:
        st.metric("Total Time", f"{stats.get('total_time', 0):.1f}s")
    with col6:
        avg_tokens = stats.get('total_tokens', 0) // max(stats.get('total_chunks', 1), 1)
        st.metric("Avg Tokens/Chunk", avg_tokens)
    
    # Display embedding info
    index_stats = st.session_state.vector_store.get_index_stats()
    st.info(f"🔢 Embedding Model: **{index_stats['provider'].upper()} - {index_stats['model']}** | Dimensions: {index_stats['embedding_dim']}")
    
    if len(st.session_state.all_crawl_stats) > 0:
        with st.expander("📋 Crawled URLs Details"):
            for i, crawl in enumerate(st.session_state.all_crawl_stats, 1):
                st.markdown(f"**{i}. {crawl['url']}**")
                st.markdown(f"   - Pages: {crawl['pages_crawled']} | Chunks: {crawl['total_chunks']} | Tokens: {crawl['total_tokens']:,}")
                st.markdown(f"   - Crawl time: {crawl['crawl_time']}s | Embed time: {crawl['embed_time']}s")

def display_chat_interface():
    """Display the chat interface with enhanced citations"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander("📚 Sources & Citations"):
                        for i, source in enumerate(message["sources"], 1):
                            relevance_pct = int(source['relevance_score'] * 100)
                            st.markdown(f"**[{i}]** [{source['title']}]({source['url']})")
                            st.progress(relevance_pct / 100, text=f"Relevance: {relevance_pct}%")
                if message.get("chunks_used"):
                    model_info = message.get("model_used", "")
                    caption = f"Used {message['chunks_used']} context chunks"
                    if model_info:
                        caption += f" • Model: {model_info}"
                    st.caption(caption)
    
    if prompt := st.chat_input("Ask a question about the website..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.generate_answer(prompt)
            
            st.markdown(response['answer'])
            
            if response.get('sources'):
                with st.expander("📚 Sources & Citations"):
                    for i, source in enumerate(response['sources'], 1):
                        relevance_pct = int(source['relevance_score'] * 100)
                        st.markdown(f"**[{i}]** [{source['title']}]({source['url']})")
                        st.progress(relevance_pct / 100, text=f"Relevance: {relevance_pct}%")
            
            if response.get('chunks_retrieved'):
                model_info = response.get('model_used', '')
                caption = f"Used {response['chunks_retrieved']} context chunks"
                if model_info:
                    caption += f" • Model: {model_info}"
                st.caption(caption)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response['answer'],
            "sources": response.get('sources', []),
            "chunks_used": response.get('chunks_retrieved', 0),
            "model_used": response.get('model_used', '')
        })

def main():
    initialize_session_state()
    
    st.title("🤖 Website RAG Chatbot")
    st.markdown("*Crawl any website and ask questions about its content*")
    
    api_keys = check_api_keys()
    
    with st.sidebar:
        st.header("🔢 Embedding Configuration")
        
        # Embedding provider selection
        embed_options = []
        if api_keys['openai']:
            embed_options.append("OpenAI")
        if api_keys['gemini']:
            embed_options.append("Gemini")
        
        if not embed_options:
            st.error("⚠️ No API keys configured!")
        else:
            current_embed = "OpenAI" if st.session_state.embedding_provider == "openai" else "Gemini"
            if current_embed not in embed_options:
                current_embed = embed_options[0]
            
            embed_provider = st.selectbox(
                "Embedding Provider",
                options=embed_options,
                index=embed_options.index(current_embed) if current_embed in embed_options else 0,
                help="Provider used to generate text embeddings for search"
            )
            
            embed_provider_key = "openai" if embed_provider == "OpenAI" else "gemini"
            
            if embed_provider_key != st.session_state.embedding_provider:
                if st.session_state.is_indexed:
                    st.warning("⚠️ Changing embedding provider will clear the current index")
                    if st.button("Confirm Change", type="primary", key="confirm_embed_change"):
                        update_embedding_provider(embed_provider_key)
                        st.rerun()
                else:
                    update_embedding_provider(embed_provider_key)
            
            # Show embedding model info
            if embed_provider_key == "openai":
                st.caption("📊 Model: text-embedding-3-small (1536 dims)")
            else:
                st.caption("📊 Model: text-embedding-004 (768 dims)")
        
        st.divider()
        
        st.header("🤖 Chat Model Configuration")
        
        # Model selection
        provider_options = []
        if api_keys['gemini']:
            provider_options.append("Gemini")
        if api_keys['openai']:
            provider_options.append("OpenAI")
        
        if not provider_options:
            st.error("⚠️ No API keys configured!")
            st.info("Add OPENAI_API_KEY or GEMINI_API_KEY to your .env file")
        else:
            provider = st.selectbox(
                "AI Provider",
                options=provider_options,
                index=0 if "Gemini" in provider_options else 0
            )
            
            if provider == "Gemini":
                model = st.selectbox(
                    "Model",
                    options=["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"],
                    index=0
                )
                provider_key = "gemini"
            else:
                model = st.selectbox(
                    "Model",
                    options=["gpt-5", "gpt-4o", "gpt-4-turbo"],
                    index=0
                )
                provider_key = "openai"
            
            # Update chatbot if model changed
            if (provider_key != st.session_state.selected_provider or 
                model != st.session_state.selected_model):
                update_chatbot_model(provider_key, model)
            
            st.caption(f"✅ Using: {provider} - {model}")
        
        st.divider()
        
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
        
        if st.session_state.is_indexed:
            crawl_mode = st.radio(
                "Crawl Mode",
                options=["Replace existing", "Add to existing"],
                index=0,
                help="Choose whether to replace or add to the current knowledge base"
            )
            append_mode = crawl_mode == "Add to existing"
        else:
            append_mode = False
        
        crawl_disabled = not (api_keys['openai'] or api_keys['gemini'])
        if st.button("🚀 Crawl & Build Knowledge Base", type="primary", use_container_width=True, disabled=crawl_disabled):
            if not url:
                st.error("Please enter a URL")
            else:
                with st.spinner("Processing..."):
                    success = crawl_and_index_website(url, max_depth, max_pages, append_mode)
                    if success:
                        st.success("Knowledge base built successfully!")
                        st.rerun()
        
        if crawl_disabled:
            st.warning("Add API key to .env file to enable crawling")
        
        st.divider()
        
        if st.session_state.crawl_stats:
            st.header("📊 Quick Stats")
            stats = st.session_state.crawl_stats
            
            st.metric("Pages Crawled", stats.get('pages_crawled', 0))
            st.metric("Text Chunks", stats.get('total_chunks', 0))
            st.metric("Total Tokens", f"{stats.get('total_tokens', 0):,}")
            st.metric("URLs Indexed", stats.get('urls_count', 0))
            
            if st.session_state.crawled_urls:
                st.caption("**Indexed URLs:**")
                for url in st.session_state.crawled_urls:
                    st.caption(f"• {url[:40]}...")
        
        st.divider()
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chatbot.clear_history()
            st.rerun()
        
        if st.button("🔄 Reset Everything", use_container_width=True):
            st.session_state.vector_store.clear()
            st.session_state.messages = []
            st.session_state.chatbot.clear_history()
            st.session_state.crawled_urls = []
            st.session_state.crawl_stats = None
            st.session_state.is_indexed = False
            st.session_state.all_crawl_stats = []
            st.rerun()
    
    if st.session_state.is_indexed:
        urls_text = ", ".join(st.session_state.crawled_urls[:3])
        if len(st.session_state.crawled_urls) > 3:
            urls_text += f" (+{len(st.session_state.crawled_urls) - 3} more)"
        st.success(f"✅ Knowledge base ready! Indexed: **{urls_text}**")
        
        tab1, tab2 = st.tabs(["💬 Chat", "📊 Statistics"])
        
        with tab1:
            display_chat_interface()
        
        with tab2:
            display_crawl_statistics()
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
        
        st.markdown("---")
        st.markdown("### Features")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("- 🔗 Crawl multiple URLs and merge knowledge bases")
            st.markdown("- 📊 Detailed crawl statistics dashboard")
            st.markdown("- 📚 Source citations with relevance scores")
        with col2:
            st.markdown("- 🤖 Support for OpenAI and Google Gemini (embeddings + chat)")
            st.markdown("- ⚡ Fast semantic search with FAISS")
            st.markdown("- 💬 Conversational chat interface")

if __name__ == "__main__":
    main()
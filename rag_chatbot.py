"""
RAG Chatbot Module
Implements the RAG pipeline: retrieve relevant chunks and generate answers using LLM.
"""

import os
from typing import List, Dict, Tuple, Optional
from openai import OpenAI

# the newest OpenAI model is "gpt-5" which was released August 7, 2025
# do not change this unless explicitly requested by the user


class RAGChatbot:
    def __init__(self, vector_store, model: str = "gpt-5"):
        self.vector_store = vector_store
        self.model = model
        self.client = None
        self.conversation_history: List[Dict] = []
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
    
    def is_configured(self) -> bool:
        """Check if OpenAI API is configured"""
        return self.client is not None
    
    def retrieve_context(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Retrieve relevant chunks for the query"""
        return self.vector_store.search(query, top_k)
    
    def format_context(self, retrieved_chunks: List[Tuple[Dict, float]]) -> str:
        """Format retrieved chunks into context string"""
        if not retrieved_chunks:
            return ""
        
        context_parts = []
        for i, (chunk, score) in enumerate(retrieved_chunks, 1):
            source = chunk.get('url', 'Unknown source')
            title = chunk.get('title', 'Untitled')
            text = chunk.get('text', '')
            
            context_parts.append(
                f"[Source {i}] {title}\n"
                f"URL: {source}\n"
                f"Content: {text}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def get_sources(self, retrieved_chunks: List[Tuple[Dict, float]]) -> List[Dict]:
        """Get source information for citations"""
        sources = []
        seen_urls = set()
        
        for chunk, score in retrieved_chunks:
            url = chunk.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    'url': url,
                    'title': chunk.get('title', 'Untitled'),
                    'relevance_score': round(score, 3)
                })
        
        return sources
    
    def generate_answer(self, query: str, top_k: int = 5, 
                        include_history: bool = True) -> Dict:
        """Generate answer using RAG pipeline"""
        if not self.client:
            return {
                'answer': "Error: OpenAI API key not configured. Please add your API key.",
                'sources': [],
                'context_used': False
            }
        
        retrieved_chunks = self.retrieve_context(query, top_k)
        
        if not retrieved_chunks:
            return {
                'answer': "I couldn't find any relevant information in the knowledge base to answer your question. Please make sure a website has been crawled and indexed first.",
                'sources': [],
                'context_used': False
            }
        
        context = self.format_context(retrieved_chunks)
        sources = self.get_sources(retrieved_chunks)
        
        system_prompt = """You are a helpful assistant that answers questions based on the provided context from a website. 
Your task is to:
1. Answer the user's question using ONLY the information provided in the context
2. Be accurate and cite which source you're using when relevant
3. If the context doesn't contain enough information to fully answer the question, say so
4. Keep your answers clear, concise, and well-organized
5. If asked about something not in the context, politely explain that the information isn't available in the crawled website content

Remember: Only use information from the provided context. Do not make up information."""

        user_message = f"""Context from the website:
{context}

---

User Question: {query}

Please answer the question based on the context provided above."""

        messages = [{"role": "system", "content": system_prompt}]
        
        if include_history and self.conversation_history:
            recent_history = self.conversation_history[-6:]
            messages.extend(recent_history)
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=1024
            )
            
            answer = response.choices[0].message.content
            
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return {
                'answer': answer,
                'sources': sources,
                'context_used': True,
                'chunks_retrieved': len(retrieved_chunks)
            }
            
        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'sources': sources,
                'context_used': True,
                'error': str(e)
            }
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_conversation_summary(self) -> Dict:
        """Get summary of current conversation"""
        return {
            'num_messages': len(self.conversation_history),
            'has_context': self.vector_store.index is not None
        }

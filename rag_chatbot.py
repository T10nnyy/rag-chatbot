"""
RAG Chatbot Module
Implements the RAG pipeline: retrieve relevant chunks and generate answers using LLM.
Supports both OpenAI (GPT) and Google Gemini models.
"""

import os
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class RAGChatbot:
    def __init__(self, vector_store, model: str = "gemini-2.0-flash", provider: str = "gemini"):
        """
        Initialize RAG Chatbot
        
        Args:
            vector_store: Vector store for retrieving relevant chunks
            model: Model name (e.g., "gpt-5", "gemini-2.0-flash-exp")
            provider: "openai" or "gemini"
        """
        self.vector_store = vector_store
        self.model = model
        self.provider = provider.lower()
        self.client = None
        self.conversation_history: List[Dict] = []
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize API client based on provider"""
        if self.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
        
        elif self.provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(self.model)
            else:
                print("Warning: GEMINI_API_KEY not found in environment variables")
    
    def is_configured(self) -> bool:
        """Check if API is configured"""
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
    
    def _generate_with_openai(self, messages: List[Dict]) -> str:
        """Generate response using OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=1024
        )
        return response.choices[0].message.content
    
    def _generate_with_gemini(self, system_prompt: str, user_message: str, 
                               include_history: bool) -> str:
        """Generate response using Gemini API"""
        # Gemini uses a different conversation format
        if include_history and self.conversation_history:
            # Build conversation history for Gemini
            chat = self.client.start_chat(history=[])
            
            # Add system instruction as first message
            full_prompt = f"{system_prompt}\n\n{user_message}"
            
            # Add conversation history
            for msg in self.conversation_history[-6:]:  # Last 3 exchanges
                if msg['role'] == 'user':
                    chat.send_message(msg['content'])
                # Assistant messages are automatically added by Gemini
            
            # Send current message
            response = chat.send_message(full_prompt)
        else:
            # No history - single message
            full_prompt = f"{system_prompt}\n\n{user_message}"
            response = self.client.generate_content(full_prompt)
        
        return response.text
    
    def generate_answer(self, query: str, top_k: int = 5, 
                        include_history: bool = True) -> Dict:
        """Generate answer using RAG pipeline"""
        if not self.client:
            return {
                'answer': f"Error: {self.provider.upper()} API key not configured. Please add your API key to the .env file.",
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

        try:
            if self.provider == "openai":
                messages = [{"role": "system", "content": system_prompt}]
                
                if include_history and self.conversation_history:
                    recent_history = self.conversation_history[-6:]
                    messages.extend(recent_history)
                
                messages.append({"role": "user", "content": user_message})
                answer = self._generate_with_openai(messages)
            
            elif self.provider == "gemini":
                answer = self._generate_with_gemini(system_prompt, user_message, include_history)
            
            else:
                return {
                    'answer': f"Error: Unsupported provider '{self.provider}'",
                    'sources': sources,
                    'context_used': False
                }
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            # Keep history manageable
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return {
                'answer': answer,
                'sources': sources,
                'context_used': True,
                'chunks_retrieved': len(retrieved_chunks),
                'model_used': f"{self.provider}/{self.model}"
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
            'has_context': self.vector_store.index is not None,
            'provider': self.provider,
            'model': self.model
        }
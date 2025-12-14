"""
Vector Store Module
Handles embedding generation with OpenAI/Gemini and vector storage/search with FAISS.
Supports multiple embedding providers for flexibility.
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class VectorStore:
    def __init__(self, embedding_provider: str = "openai", embedding_model: str = None):
        """
        Initialize Vector Store
        
        Args:
            embedding_provider: "openai" or "gemini"
            embedding_model: Model name (auto-selected if None)
        """
        self.embedding_provider = embedding_provider.lower()
        
        # Set default models based on provider
        if embedding_model is None:
            if self.embedding_provider == "openai":
                self.embedding_model = "text-embedding-3-small"
            elif self.embedding_provider == "gemini":
                self.embedding_model = "models/text-embedding-004"
            else:
                raise ValueError(f"Unsupported provider: {embedding_provider}")
        else:
            self.embedding_model = embedding_model
        
        # Set embedding dimensions based on provider and model
        if self.embedding_provider == "openai":
            if "text-embedding-3-small" in self.embedding_model:
                self.embedding_dim = 1536
            elif "text-embedding-3-large" in self.embedding_model:
                self.embedding_dim = 3072
            else:
                self.embedding_dim = 1536  # Default
        elif self.embedding_provider == "gemini":
            self.embedding_dim = 768  # Gemini text-embedding-004 uses 768 dimensions
        
        self.index = None
        self.chunks: List[Dict] = []
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize API client based on provider"""
        if self.embedding_provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
        
        elif self.embedding_provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.client = True  # Flag to indicate configured
    
    def is_configured(self) -> bool:
        """Check if API is configured"""
        return self.client is not None
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if not self.client:
            raise ValueError(f"{self.embedding_provider.upper()} API key not configured")
        
        text = text.replace("\n", " ").strip()
        if not text:
            return np.zeros(self.embedding_dim)
        
        if self.embedding_provider == "openai":
            return self._generate_embedding_openai(text)
        elif self.embedding_provider == "gemini":
            return self._generate_embedding_gemini(text)
        else:
            raise ValueError(f"Unsupported provider: {self.embedding_provider}")
    
    def _generate_embedding_openai(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI"""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    def _generate_embedding_gemini(self, text: str) -> np.ndarray:
        """Generate embedding using Gemini"""
        result = genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type="retrieval_document"
        )
        return np.array(result['embedding'], dtype=np.float32)
    
    def generate_embeddings_batch(self, texts: List[str], 
                                   progress_callback=None) -> np.ndarray:
        """Generate embeddings for multiple texts in batches"""
        if not self.client:
            raise ValueError(f"{self.embedding_provider.upper()} API key not configured")
        
        if self.embedding_provider == "openai":
            return self._generate_embeddings_batch_openai(texts, progress_callback)
        elif self.embedding_provider == "gemini":
            return self._generate_embeddings_batch_gemini(texts, progress_callback)
        else:
            raise ValueError(f"Unsupported provider: {self.embedding_provider}")
    
    def _generate_embeddings_batch_openai(self, texts: List[str], 
                                          progress_callback=None) -> np.ndarray:
        """Generate embeddings using OpenAI (batch processing)"""
        all_embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            cleaned_batch = [t.replace("\n", " ").strip() for t in batch]
            
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=cleaned_batch
            )
            
            batch_embeddings = [np.array(item.embedding, dtype=np.float32) 
                               for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            if progress_callback:
                progress_callback(min(i + batch_size, len(texts)), len(texts))
        
        return np.array(all_embeddings)
    
    def _generate_embeddings_batch_gemini(self, texts: List[str], 
                                          progress_callback=None) -> np.ndarray:
        """Generate embeddings using Gemini (batch processing)"""
        all_embeddings = []
        batch_size = 100  # Gemini supports batch embedding
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            cleaned_batch = [t.replace("\n", " ").strip() for t in batch]
            
            # Gemini can handle batches directly
            result = genai.embed_content(
                model=self.embedding_model,
                content=cleaned_batch,
                task_type="retrieval_document"
            )
            
            # Handle both single and batch responses
            if isinstance(result['embedding'][0], list):
                # Batch response
                batch_embeddings = [np.array(emb, dtype=np.float32) 
                                   for emb in result['embedding']]
            else:
                # Single response
                batch_embeddings = [np.array(result['embedding'], dtype=np.float32)]
            
            all_embeddings.extend(batch_embeddings)
            
            if progress_callback:
                progress_callback(min(i + batch_size, len(texts)), len(texts))
        
        return np.array(all_embeddings)
    
    def build_index(self, chunks: List[Dict], progress_callback=None) -> bool:
        """Build FAISS index from chunks"""
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS not installed. Please install faiss-cpu.")
        
        if not chunks:
            return False
        
        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]
        
        embeddings = self.generate_embeddings_batch(texts, progress_callback)
        
        self.embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        return True
    
    def add_chunks(self, chunks: List[Dict], progress_callback=None) -> bool:
        """Add new chunks to existing index (for multiple URL support)"""
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS not installed. Please install faiss-cpu.")
        
        if not chunks:
            return False
        
        if self.index is None:
            return self.build_index(chunks, progress_callback)
        
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.generate_embeddings_batch(texts, progress_callback)
        
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        
        return True
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for most relevant chunks"""
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS not installed.")
        
        if self.index is None or not self.chunks:
            return []
        
        # Generate query embedding with appropriate task type for Gemini
        if self.embedding_provider == "gemini":
            result = genai.embed_content(
                model=self.embedding_model,
                content=query.replace("\n", " ").strip(),
                task_type="retrieval_query"  # Use query task type for search
            )
            query_embedding = np.array(result['embedding'], dtype=np.float32)
        else:
            query_embedding = self.generate_embedding(query)
        
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the index"""
        if self.index is None:
            return {
                'indexed': False,
                'num_vectors': 0,
                'embedding_dim': self.embedding_dim,
                'num_chunks': 0,
                'provider': self.embedding_provider,
                'model': self.embedding_model
            }
        
        return {
            'indexed': True,
            'num_vectors': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'num_chunks': len(self.chunks),
            'provider': self.embedding_provider,
            'model': self.embedding_model
        }
    
    def clear(self):
        """Clear the index and chunks"""
        self.index = None
        self.chunks = []
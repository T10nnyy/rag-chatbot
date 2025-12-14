"""
Vector Store Module
Handles embedding generation with OpenAI and vector storage/search with FAISS.
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from openai import OpenAI

# the newest OpenAI model is "gpt-5" which was released August 7, 2025
# do not change this unless explicitly requested by the user


class VectorStore:
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embedding_model = embedding_model
        self.embedding_dim = 1536
        self.index = None
        self.chunks: List[Dict] = []
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
    
    def is_configured(self) -> bool:
        """Check if OpenAI API is configured"""
        return self.client is not None
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if not self.client:
            raise ValueError("OpenAI API key not configured")
        
        text = text.replace("\n", " ").strip()
        if not text:
            return np.zeros(self.embedding_dim)
        
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    def generate_embeddings_batch(self, texts: List[str], 
                                   progress_callback=None) -> np.ndarray:
        """Generate embeddings for multiple texts in batches"""
        if not self.client:
            raise ValueError("OpenAI API key not configured")
        
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
                'num_chunks': 0
            }
        
        return {
            'indexed': True,
            'num_vectors': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'num_chunks': len(self.chunks)
        }
    
    def clear(self):
        """Clear the index and chunks"""
        self.index = None
        self.chunks = []

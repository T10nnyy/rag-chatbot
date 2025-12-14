"""
Text Processing Module
Handles text preprocessing, cleaning, and chunking for the RAG pipeline.
"""

import re
from typing import List, Dict
import tiktoken


class TextProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        return text
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_chunks(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Create overlapping chunks from text using token-based splitting.
        Each chunk includes metadata about its source.
        """
        if not text or len(text.strip()) < 50:
            return []
        
        cleaned_text = self.clean_text(text)
        sentences = self.split_into_sentences(cleaned_text)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            if sentence_tokens > self.chunk_size:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(self._create_chunk_dict(chunk_text, metadata, len(chunks)))
                    current_chunk = []
                    current_tokens = 0
                
                words = sentence.split()
                temp_chunk = []
                temp_tokens = 0
                
                for word in words:
                    word_tokens = self.count_tokens(word + ' ')
                    if temp_tokens + word_tokens > self.chunk_size:
                        if temp_chunk:
                            chunk_text = ' '.join(temp_chunk)
                            chunks.append(self._create_chunk_dict(chunk_text, metadata, len(chunks)))
                        temp_chunk = [word]
                        temp_tokens = word_tokens
                    else:
                        temp_chunk.append(word)
                        temp_tokens += word_tokens
                
                if temp_chunk:
                    current_chunk = temp_chunk
                    current_tokens = temp_tokens
                continue
            
            if current_tokens + sentence_tokens > self.chunk_size:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(self._create_chunk_dict(chunk_text, metadata, len(chunks)))
                    
                    overlap_tokens = 0
                    overlap_start = len(current_chunk)
                    
                    for i in range(len(current_chunk) - 1, -1, -1):
                        sent_tokens = self.count_tokens(current_chunk[i])
                        if overlap_tokens + sent_tokens <= self.chunk_overlap:
                            overlap_tokens += sent_tokens
                            overlap_start = i
                        else:
                            break
                    
                    current_chunk = current_chunk[overlap_start:] + [sentence]
                    current_tokens = sum(self.count_tokens(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(self._create_chunk_dict(chunk_text, metadata, len(chunks)))
        
        return chunks
    
    def _create_chunk_dict(self, text: str, metadata: Dict, chunk_index: int) -> Dict:
        """Create a chunk dictionary with text and metadata"""
        chunk = {
            'text': text,
            'chunk_index': chunk_index,
            'token_count': self.count_tokens(text)
        }
        if metadata:
            chunk.update({
                'url': metadata.get('url', ''),
                'title': metadata.get('title', ''),
                'source': metadata.get('url', '')
            })
        return chunk
    
    def process_crawled_data(self, crawled_pages: List[Dict]) -> List[Dict]:
        """
        Process all crawled pages and create chunks.
        Each chunk includes source URL and title for citation.
        """
        all_chunks = []
        
        for page in crawled_pages:
            content = page.get('content', '')
            headings = page.get('headings', [])
            
            heading_text = '\n'.join([h['text'] for h in headings])
            full_content = f"{heading_text}\n\n{content}" if heading_text else content
            
            metadata = {
                'url': page.get('url', ''),
                'title': page.get('title', '')
            }
            
            page_chunks = self.create_chunks(full_content, metadata)
            all_chunks.extend(page_chunks)
        
        return all_chunks
    
    def get_processing_stats(self, chunks: List[Dict]) -> Dict:
        """Get statistics about processed chunks"""
        if not chunks:
            return {
                'total_chunks': 0,
                'total_tokens': 0,
                'avg_tokens_per_chunk': 0,
                'unique_sources': 0
            }
        
        total_tokens = sum(c.get('token_count', 0) for c in chunks)
        unique_sources = len(set(c.get('url', '') for c in chunks))
        
        return {
            'total_chunks': len(chunks),
            'total_tokens': total_tokens,
            'avg_tokens_per_chunk': total_tokens // len(chunks) if chunks else 0,
            'unique_sources': unique_sources
        }

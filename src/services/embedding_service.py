from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List
from src.models.schemas import DocumentChunk, ClauseMatch

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.document_store = []
    
    def embed_documents(self, chunks: List[DocumentChunk]) -> None:
        """Generate embeddings for document chunks"""
        if not chunks:
            return
            
        texts = [chunk.content for chunk in chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        
        # Store in FAISS
        self.index.add(embeddings.astype('float32'))
        self.document_store.extend(chunks)
        
        print(f"✅ Embedded {len(chunks)} document chunks")
    
    def search_similar_chunks(self, query: str, top_k: int = 5) -> List[ClauseMatch]:
        """Search for similar document chunks"""
        if len(self.document_store) == 0:
            return []
            
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search in FAISS
        scores, indices = self.index.search(
            query_embedding.astype('float32'), min(top_k, len(self.document_store))
        )
        
        matches = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.document_store) and idx >= 0:
                chunk = self.document_store[idx]
                matches.append(ClauseMatch(
                    content=chunk.content,
                    similarity_score=float(score),
                    source_document=chunk.metadata.get('source', 'unknown'),
                    page_number=chunk.metadata.get('page_number')
                ))
        
        return matches

"""
High-Accuracy HackRx LLM-Powered Query Retrieval System (Memory Optimized)
A robust, generalized RAG system designed for maximum accuracy on low-memory environments.
Run with: uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os
import io
import time
import requests
import asyncio
from typing import List, Dict, Any
from urllib.parse import urlparse
import numpy as np
import re

# FastAPI and HTTP
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Document processing
import PyPDF2
from docx import Document

# AI/ML
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss

# Load environment variables from a .env file
load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
#                                 CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HACKRX_TOKEN = os.getenv("HACKRX_API_KEY")

# Validate that the necessary API keys are set on startup
if not GEMINI_API_KEY:
    raise ValueError("FATAL ERROR: GEMINI_API_KEY environment variable is not set.")
if not HACKRX_TOKEN:
    raise ValueError("FATAL ERROR: HACKRX_API_KEY environment variable is not set.")

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
LLM_MODEL = genai.GenerativeModel('gemini-1.5-flash')
GENERATION_CONFIG = genai.types.GenerationConfig(
    candidate_count=1,
    max_output_tokens=400,
    temperature=0.0,
)

# --- MEMORY OPTIMIZATION: Lazy Loading for Embedding Model ---
EMBEDDING_MODEL = None
EMBEDDING_DIM = 384

def get_embedding_model():
    """Lazily loads the SentenceTransformer model to save memory on startup."""
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        print("🧠 Lazily loading embedding model (this will happen only once)...")
        EMBEDDING_MODEL = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
        print("✅ Embedding model loaded into memory.")
    return EMBEDDING_MODEL

# FAISS Index and Document Store
FAISS_INDEX = faiss.IndexFlatL2(EMBEDDING_DIM)
DOCUMENT_STORE = []

# FastAPI App
app = FastAPI(
    title="HackRx LLM Query Retrieval System",
    description="A generalized RAG system optimized for competition deployment.",
    version="4.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# ═══════════════════════════════════════════════════════════════════════════════
#                                 DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class DocumentChunk(BaseModel):
    content: str
    metadata: Dict[str, Any]

class ClauseMatch(BaseModel):
    content: str
    similarity_score: float

# ═══════════════════════════════════════════════════════════════════════════════
#                           DOCUMENT PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

class DocumentProcessor:
    def __init__(self):
        self.chunk_size = 300  # Slightly larger chunks for better context
    
    def download_document(self, url: str) -> bytes:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, timeout=45, headers=headers)
            response.raise_for_status()
            print(f"✅ Downloaded document: {len(response.content)} bytes")
            return response.content
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download document from URL: {str(e)}")
    
    def get_file_type(self, url: str) -> str:
        parsed_url = urlparse(url)
        file_path = parsed_url.path.lower()
        if file_path.endswith('.pdf'): return 'pdf'
        if file_path.endswith('.docx'): return 'docx'
        # Default to PDF for blob URLs or unknown types
        return 'pdf'
    
    def extract_text(self, content: bytes, file_type: str) -> str:
        if file_type == 'pdf': return self._extract_text_from_pdf(content)
        if file_type == 'docx': return self._extract_text_from_docx(content)
        raise ValueError(f"Unsupported file type: {file_type}")

    def _extract_text_from_pdf(self, content: bytes) -> str:
        try:
            text = ""
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            for page_num, page in enumerate(reader.pages):
                if page_text := page.extract_text(): 
                    # Clean and normalize the text
                    page_text = re.sub(r'\s+', ' ', page_text.strip())
                    text += page_text + "\n"
            print(f"📄 Extracted {len(text)} characters from PDF ({len(reader.pages)} pages).")
            return text
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def _extract_text_from_docx(self, content: bytes) -> str:
        try:
            doc = Document(io.BytesIO(content))
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            # Clean and normalize the text
            text = re.sub(r'\s+', ' ', text.strip())
            print(f"📄 Extracted {len(text)} characters from DOCX.")
            return text
        except Exception as e:
            raise Exception(f"Failed to extract text from DOCX: {str(e)}")
    
    def sentence_aware_chunking(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        # Enhanced sentence splitting that handles more cases
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.replace('\n', ' '))
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        chunks, current_chunk_sentences, current_chunk_length = [], [], 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_chunk_length + sentence_length <= self.chunk_size:
                current_chunk_sentences.append(sentence)
                current_chunk_length += sentence_length
            else:
                if current_chunk_sentences:
                    chunk_content = " ".join(current_chunk_sentences)
                    chunks.append(DocumentChunk(
                        content=chunk_content, 
                        metadata={"chunk_id": len(chunks), **metadata}
                    ))
                
                current_chunk_sentences = [sentence]
                current_chunk_length = sentence_length
        
        # Add the last chunk if it exists
        if current_chunk_sentences:
            chunk_content = " ".join(current_chunk_sentences)
            chunks.append(DocumentChunk(
                content=chunk_content, 
                metadata={"chunk_id": len(chunks), **metadata}
            ))
        
        print(f"📊 Created {len(chunks)} sentence-aware chunks.")
        return chunks

# ═══════════════════════════════════════════════════════════════════════════════
#                           SEMANTIC SEARCH SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class SemanticSearchService:
    def __init__(self):
        self.faiss_index = FAISS_INDEX
        self.document_store = DOCUMENT_STORE
    
    def embed_and_index(self, chunks: List[DocumentChunk], batch_size: int = 16) -> None:
        """
        Embeds chunks in smaller batches to prevent memory overload on low-resource servers.
        """
        if not chunks: return
        
        # Reset index and store
        self.faiss_index.reset()
        self.document_store.clear()
        
        model = get_embedding_model()
        print(f"🧠 Generating embeddings for {len(chunks)} chunks in batches of {batch_size}...")
        
        # Process in smaller batches
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_texts = [chunk.content for chunk in batch_chunks]
            
            print(f"   - Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}...")
            
            try:
                embeddings = model.encode(batch_texts, normalize_embeddings=True, batch_size=8)
                self.faiss_index.add(embeddings.astype('float32'))
                
                # Store document metadata
                for chunk in batch_chunks:
                    self.document_store.append({
                        "content": chunk.content, 
                        "metadata": chunk.metadata
                    })
                    
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                continue

        print(f"✅ Embedded and indexed {self.faiss_index.ntotal} chunks.")
    
    async def search(self, query: str, top_k: int = 10) -> List[ClauseMatch]:
        if self.faiss_index.ntotal == 0: 
            return []
        
        try:
            model = get_embedding_model()
            query_embedding = model.encode([query], normalize_embeddings=True)
            
            # Search for relevant chunks
            k = min(top_k, self.faiss_index.ntotal)
            distances, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1 and idx < len(self.document_store):
                    results.append(ClauseMatch(
                        content=self.document_store[idx]["content"], 
                        similarity_score=float(dist)
                    ))
            
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []

# ═══════════════════════════════════════════════════════════════════════════════
#                               LLM SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class LLMService:
    def __init__(self):
        self.model = LLM_MODEL
        self.generation_config = GENERATION_CONFIG
    
    def create_high_accuracy_prompt(self, query: str, relevant_chunks: List[ClauseMatch]) -> str:
        # Sort chunks by similarity score (lower is better for L2 distance)
        sorted_chunks = sorted(relevant_chunks, key=lambda x: x.similarity_score)
        context = "\n\n---\n\n".join([chunk.content for chunk in sorted_chunks[:8]])
        
        return f"""**Role:** You are an expert document analyst specializing in insurance, legal, and compliance document analysis. Provide precise, factual answers based strictly on the provided document content.

**Context Document:**
---
{context}
---

**Instructions:**
1. Analyze the entire context carefully for information related to the question.
2. Provide a complete, accurate answer based ONLY on the information explicitly stated in the document.
3. Include specific details, numbers, time periods, and conditions mentioned in the document.
4. If multiple related facts exist, combine them into a comprehensive answer.
5. If the information is not available in the document, state this clearly.
6. Do NOT add external knowledge or assumptions.

**Question:** {query}

**Answer:**"""
    
    async def generate_answer(self, query: str, relevant_chunks: List[ClauseMatch]) -> str:
        if not relevant_chunks: 
            return "The provided document does not contain information relevant to this question."
        
        prompt = self.create_high_accuracy_prompt(query, relevant_chunks)
        
        for attempt in range(3):
            try:
                response = await self.model.generate_content_async(
                    prompt, 
                    generation_config=self.generation_config
                )
                
                if response and response.text:
                    # Clean up the response
                    answer = response.text.strip()
                    answer = re.sub(r'\s+', ' ', answer)
                    return answer
                else:
                    return "Unable to generate a response for this question."
                    
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    wait_time = (2 ** attempt) + np.random.uniform(0, 1)
                    print(f"Rate limit hit. Retrying in {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"LLM Error for '{query[:30]}...': {e}")
                    if attempt == 2:
                        return "An error occurred while generating the answer."
        
        return "Failed to generate an answer after multiple retries."
    
    async def process_all_queries(self, queries: List[str], search_service: SemanticSearchService) -> List[str]:
        """
        Processes queries with optimized performance for competition environment.
        """
        answers = []
        
        for i, query in enumerate(queries):
            print(f"🔍 Processing Q{i+1}/{len(queries)}: '{query[:50]}...'")
            
            try:
                relevant_chunks = await search_service.search(query, top_k=10)
                answer = await self.generate_answer(query, relevant_chunks)
                answers.append(answer)
                
                # Small delay to prevent overwhelming the API
                if i < len(queries) - 1:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                print(f"Error processing query {i+1}: {e}")
                answers.append("An error occurred while processing this question.")
        
        return answers

# ═══════════════════════════════════════════════════════════════════════════════
#                                 API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

doc_processor = DocumentProcessor()
search_service = SemanticSearchService()
llm_service = LLMService()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme.lower() != "bearer" or credentials.credentials != HACKRX_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_submission(request: QueryRequest, _: str = Depends(verify_token)):
    start_time = time.time()
    
    try:
        print(f"🎯 Starting RAG process for {request.documents}")
        print(f"📝 Processing {len(request.questions)} questions")
        
        # Download and process document
        content = doc_processor.download_document(request.documents)
        file_type = doc_processor.get_file_type(request.documents)
        text = doc_processor.extract_text(content, file_type)
        
        if not text.strip():
            raise Exception("No text could be extracted from the document")
        
        # Create semantic chunks
        chunks = doc_processor.sentence_aware_chunking(text, {"source": request.documents})
        
        if not chunks:
            raise Exception("No valid chunks could be created from the document")
        
        # Index chunks for semantic search
        search_service.embed_and_index(chunks)
        
        # Process all questions
        answers = await llm_service.process_all_queries(request.questions, search_service)
        
        processing_time = time.time() - start_time
        print(f"✅ Process completed in {processing_time:.2f}s.")
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Critical error during RAG process: {error_msg}")
        raise HTTPException(
            status_code=500, 
            detail=f"An internal processing error occurred: {error_msg}"
        )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "version": "4.0.0", 
        "model": "gemini-1.5-flash",
        "embedding_model": "multi-qa-MiniLM-L6-cos-v1",
        "timestamp": time.time()
    }

@app.get("/")
async def root():
    return {
        "message": "HackRx LLM Query Retrieval System",
        "version": "4.0.0",
        "endpoints": {
            "main": "/hackrx/run",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
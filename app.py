"""
High-Accuracy HackRx LLM-Powered Query Retrieval System
A robust, generalized RAG system designed for maximum accuracy.
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
    max_output_tokens=300, # Increased slightly for more comprehensive single sentences
    temperature=0.0,      # Set to 0 for maximum factuality and determinism
)

# Initialize Embedding Model & FAISS (in-memory)
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_DIM = 384
FAISS_INDEX = faiss.IndexFlatL2(EMBEDDING_DIM) # Using L2 distance for similarity
DOCUMENT_STORE = []

# FastAPI App
app = FastAPI(
    title="High-Accuracy HackRx LLM Query Retrieval System",
    description="A generalized RAG system focused on accuracy and comprehensive answers.",
    version="3.0.0"
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
        self.chunk_size = 256  # Optimal size in tokens for the embedding model
    
    def download_document(self, url: str) -> bytes:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            print(f"✅ Downloaded document: {len(response.content)} bytes")
            return response.content
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download document from URL: {str(e)}")
    
    def get_file_type(self, url: str) -> str:
        parsed_url = urlparse(url)
        file_path = parsed_url.path.lower()
        if file_path.endswith('.pdf'):
            return 'pdf'
        elif file_path.endswith('.docx'):
            return 'docx'
        return 'pdf' # Default to PDF
    
    def extract_text(self, content: bytes, file_type: str) -> str:
        if file_type == 'pdf':
            return self._extract_text_from_pdf(content)
        elif file_type == 'docx':
            return self._extract_text_from_docx(content)
        raise ValueError(f"Unsupported file type: {file_type}")

    def _extract_text_from_pdf(self, content: bytes) -> str:
        try:
            pdf_file = io.BytesIO(content)
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            print(f"📄 Extracted {len(text)} characters from PDF.")
            return text
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def _extract_text_from_docx(self, content: bytes) -> str:
        try:
            doc_file = io.BytesIO(content)
            doc = Document(doc_file)
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            print(f"📄 Extracted {len(text)} characters from DOCX.")
            return text
        except Exception as e:
            raise Exception(f"Failed to extract text from DOCX: {str(e)}")
    
    def sentence_aware_chunking(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        """
        Chunks text by grouping complete sentences, which is superior to fixed-size chunks.
        This preserves the semantic meaning of the text.
        """
        # Split the text into sentences using regex
        sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' '))
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk_sentences = []
        current_chunk_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_chunk_length + sentence_length <= self.chunk_size:
                current_chunk_sentences.append(sentence)
                current_chunk_length += sentence_length
            else:
                # Create a chunk from the current sentences
                chunk_text = " ".join(current_chunk_sentences)
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_id"] = len(chunks)
                chunks.append(DocumentChunk(content=chunk_text, metadata=chunk_metadata))
                
                # Start a new chunk with the current sentence
                current_chunk_sentences = [sentence]
                current_chunk_length = sentence_length
        
        # Add the last remaining chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = len(chunks)
            chunks.append(DocumentChunk(content=chunk_text, metadata=chunk_metadata))

        print(f"📊 Created {len(chunks)} sentence-aware chunks.")
        return chunks

# ═══════════════════════════════════════════════════════════════════════════════
#                           SEMANTIC SEARCH SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class SemanticSearchService:
    def __init__(self):
        self.embedding_model = EMBEDDING_MODEL
        self.faiss_index = FAISS_INDEX
        self.document_store = DOCUMENT_STORE
    
    def embed_and_index(self, chunks: List[DocumentChunk]) -> None:
        """Embeds document chunks and indexes them in FAISS for fast retrieval."""
        if not chunks:
            print("⚠️ No chunks to embed.")
            return
        
        self.faiss_index.reset()
        self.document_store.clear()
        
        texts = [chunk.content for chunk in chunks]
        print(f"🧠 Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        
        self.faiss_index.add(embeddings.astype('float32'))
        self.document_store.extend([{"content": chunk.content, "metadata": chunk.metadata} for chunk in chunks])
        
        print(f"✅ Embedded and indexed {self.faiss_index.ntotal} chunks.")
    
    async def search(self, query: str, top_k: int = 8) -> List[ClauseMatch]:
        """Performs a pure semantic search to find the most relevant chunks."""
        if self.faiss_index.ntotal == 0:
            return []
        
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        distances, indices = self.faiss_index.search(query_embedding.astype('float32'), min(top_k, self.faiss_index.ntotal))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                doc = self.document_store[idx]
                results.append(ClauseMatch(
                    content=doc["content"],
                    similarity_score=float(dist),
                ))
        return results

# ═══════════════════════════════════════════════════════════════════════════════
#                               LLM SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class LLMService:
    def __init__(self):
        self.model = LLM_MODEL
        self.generation_config = GENERATION_CONFIG
    
    def create_high_accuracy_prompt(self, query: str, relevant_chunks: List[ClauseMatch]) -> str:
        """
        Creates a highly-specific prompt designed to force the LLM to be factual,
        comprehensive, and concise.
        """
        context = "\n\n---\n\n".join([chunk.content for chunk in relevant_chunks])
        
        return f"""
        **Role:** You are an expert document analyst. Your task is to answer a question with extreme precision based *only* on the provided text.

        **Source Text:**
        ---
        {context}
        ---

        **Instructions:**
        1.  Analyze the entire source text provided above.
        2.  Identify all facts, figures, and conditions directly related to the user's question.
        3.  Synthesize these facts into a single, comprehensive sentence.
        4.  Your answer **MUST** be a single sentence.
        5.  Do **NOT** add any information that is not explicitly stated in the source text.
        6.  Do **NOT** start your answer with phrases like "According to the document..." or "The answer is...". Answer directly.
        7.  Give the answer such that you are giving a examination answer and give single word answers in statement (like "average lifespan of humans is 65yrs" instead of just "65yrs").

        **User Question:** {query}

        **Single-Sentence Answer:**
        """
    
    async def generate_answer(self, query: str, relevant_chunks: List[ClauseMatch]) -> str:
        """Generates a high-quality answer with an exponential backoff retry mechanism."""
        if not relevant_chunks:
            return "The provided document does not contain information relevant to this question."
        
        prompt = self.create_high_accuracy_prompt(query, relevant_chunks)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.model.generate_content_async(
                    prompt,
                    generation_config=self.generation_config
                )
                answer = response.text.strip().replace('\n', ' ')
                return ' '.join(answer.split())
            except Exception as e:
                error_str = str(e)
                if "429" in error_str and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + np.random.uniform(0, 1)
                    print(f"Rate limit hit (429). Retrying in {wait_time:.2f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"❌ LLM Error for '{query[:30]}...': {error_str}")
                    return f"An error occurred while generating the answer."
        return "Failed to generate an answer after multiple retries due to API issues."
    
    async def process_all_queries(self, queries: List[str], search_service: SemanticSearchService) -> List[str]:
        """Processes a list of queries concurrently for maximum efficiency."""
        async def _process_one(query: str):
            print(f"🔍 Searching for context for Q: '{query[:50]}...'")
            # Retrieve more chunks to give the LLM a wider context
            relevant_chunks = await search_service.search(query, top_k=8) 
            if not relevant_chunks:
                print(f"⚠️ No relevant chunks found for Q: '{query[:50]}...'")
            return await self.generate_answer(query, relevant_chunks)

        tasks = [_process_one(q) for q in queries]
        answers = await asyncio.gather(*tasks)
        return answers

# ═══════════════════════════════════════════════════════════════════════════════
#                                 API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

# Initialize services
doc_processor = DocumentProcessor()
search_service = SemanticSearchService()
llm_service = LLMService()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme.lower() != "bearer" or credentials.credentials != HACKRX_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_submission(request: QueryRequest, _: str = Depends(verify_token)):
    """Main endpoint to process a document and answer questions with high accuracy."""
    start_time = time.time()
    try:
        print(f"🎯 Starting high-accuracy RAG process for {request.documents}")
        
        # Step 1: Download and Extract Text
        content = doc_processor.download_document(request.documents)
        file_type = doc_processor.get_file_type(request.documents)
        text = doc_processor.extract_text(content, file_type)
        
        # Step 2: Perform Sentence-Aware Chunking
        metadata = {"source": request.documents}
        chunks = doc_processor.sentence_aware_chunking(text, metadata)
        
        # Step 3: Embed and Index all chunks
        search_service.embed_and_index(chunks)
        
        # Step 4: Concurrently process all questions
        answers = await llm_service.process_all_queries(request.questions, search_service)
        
        print(f"✅ Process completed in {time.time() - start_time:.2f}s. Returning {len(answers)} answers.")
        return QueryResponse(answers=answers)
        
    except Exception as e:
        print(f"❌ A critical error occurred during the RAG process: {str(e)}")
        # In a real-world scenario, you might want to log the full traceback here
        raise HTTPException(status_code=500, detail=f"An internal processing error occurred: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint to verify service status."""
    return {
        "status": "healthy",
        "version": "3.0.0",
        "model": "gemini-1.5-flash",
        "embedding_model": "all-MiniLM-L6-v2",
        "faiss_indexed_items": FAISS_INDEX.ntotal
    }

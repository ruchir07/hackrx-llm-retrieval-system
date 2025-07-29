"""
Enhanced HackRx LLM-Powered Query Retrieval System
Combines working FastAPI approach with winning Mercury AI features
Run with: uvicorn app:app --reload --host localhost --port 8000
"""

import os
import io
import time
import requests
import asyncio
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import numpy as np

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

# Load environment variables
load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
#                                CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HACKRX_TOKEN = os.getenv("HACKRX_TEAM_TOKEN")

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
LLM_MODEL = genai.GenerativeModel('gemini-1.5-flash')
GENERATION_CONFIG = genai.types.GenerationConfig(
    candidate_count=1,
    max_output_tokens=400,
    temperature=0.05,
)

# Initialize Embedding Model & FAISS
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_DIM = 384
FAISS_INDEX = faiss.IndexFlatIP(EMBEDDING_DIM)
DOCUMENT_STORE = []

# FastAPI App
app = FastAPI(
    title="Enhanced HackRx LLM Query Retrieval System",
    description="Multi-layer search with advanced document processing",
    version="2.0.0"
)
security = HTTPBearer()

# ═══════════════════════════════════════════════════════════════════════════════
#                                DATA MODELS
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
    source_document: str
    chunk_type: str = "semantic"

# ═══════════════════════════════════════════════════════════════════════════════
#                           ENHANCED DOCUMENT PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancedDocumentProcessor:
    def __init__(self):
        self.chunk_size = 800
        self.chunk_overlap = 150
    
    def download_document(self, url: str) -> bytes:
        """Download document with better error handling"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            print(f"✅ Downloaded document: {len(response.content)} bytes")
            return response.content
        except Exception as e:
            raise Exception(f"Failed to download document: {str(e)}")
    
    def get_file_type(self, url: str, content: bytes = None) -> str:
        """Enhanced file type detection"""
        parsed_url = urlparse(url)
        file_path = parsed_url.path.lower()
        
        if file_path.endswith('.pdf'):
            return 'pdf'
        elif file_path.endswith('.docx'):
            return 'docx'
        
        # Content-based detection
        if content:
            if content.startswith(b'%PDF'):
                return 'pdf'
            elif content.startswith(b'PK\x03\x04') and b'word/' in content:
                return 'docx'
        
        return 'pdf'  # Default
    
    def extract_text_from_pdf(self, content: bytes) -> str:
        """Enhanced PDF text extraction"""
        try:
            pdf_file = io.BytesIO(content)
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n[Page {page_num + 1}]\n{page_text}\n"
                except Exception as e:
                    print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
                    continue
            
            # Clean text
            text = text.replace('\n\n\n', '\n\n')
            print(f"📄 Extracted {len(text)} characters from {len(reader.pages)} pages")
            return text
            
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def extract_text_from_docx(self, content: bytes) -> str:
        """Enhanced DOCX text extraction"""
        try:
            doc_file = io.BytesIO(content)
            doc = Document(doc_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            print(f"📄 Extracted {len(text)} characters from DOCX")
            return text
        except Exception as e:
            raise Exception(f"Failed to extract text from DOCX: {str(e)}")
    
    def intelligent_chunk_text(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        """WINNING FEATURE: Intelligent chunking with multiple strategies"""
        chunks = []
        
        # Strategy 1: Section-based chunking (NEW)
        section_chunks = self._chunk_by_sections(text, metadata)
        if len(section_chunks) > 3:
            chunks.extend(section_chunks)
            print(f"📊 Created {len(section_chunks)} section-based chunks")
        
        # Strategy 2: Sliding window with sentence awareness (ENHANCED)
        sliding_chunks = self._sliding_window_chunk(text, metadata)
        chunks.extend(sliding_chunks)
        print(f"📊 Created {len(sliding_chunks)} sliding window chunks")
        
        # Strategy 3: Keyword-focused chunks (NEW)
        keyword_chunks = self._keyword_focused_chunks(text, metadata)
        chunks.extend(keyword_chunks)
        print(f"📊 Created {len(keyword_chunks)} keyword-focused chunks")
        
        return chunks
    
    def _chunk_by_sections(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        """NEW: Split by document sections"""
        import re
        
        section_patterns = [
            r'\n\d+\.\s+[A-Z][A-Za-z\s]+',  # "1. SECTION NAME"
            r'\n[A-Z][A-Z\s]{5,}:',         # "SECTION NAME:"
            r'\nSection \d+',                # "Section 1"
            r'\nArticle \d+',                # "Article 1"
        ]
        
        for pattern in section_patterns:
            sections = re.split(pattern, text)
            if len(sections) > 3:
                chunks = []
                for i, section in enumerate(sections):
                    if len(section.strip()) > 200:
                        chunk_metadata = metadata.copy()
                        chunk_metadata.update({
                            "chunk_id": i,
                            "chunk_type": "section",
                            "section_number": i
                        })
                        chunks.append(DocumentChunk(
                            content=section.strip(),
                            metadata=chunk_metadata
                        ))
                return chunks[:15]  # Limit sections
        
        return []
    
    def _sliding_window_chunk(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        """ENHANCED: Sentence-aware sliding window"""
        chunks = []
        sentences = text.replace('\n', ' ').split('. ')
        
        current_chunk = ""
        sentence_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if current_chunk:
                current_chunk += ". " + sentence
            else:
                current_chunk = sentence
            
            sentence_count += 1
            
            # Create chunk when we have enough content
            if len(current_chunk) >= self.chunk_size or sentence_count >= 8:
                if current_chunk.strip():
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk_id": len(chunks),
                        "chunk_type": "sliding_window",
                        "sentence_count": sentence_count
                    })
                    
                    chunks.append(DocumentChunk(
                        content=current_chunk.strip(),
                        metadata=chunk_metadata
                    ))
                
                # Overlap: keep last 2 sentences
                overlap_sentences = current_chunk.split('. ')[-2:]
                if len(overlap_sentences) >= 2:
                    current_chunk = '. '.join(overlap_sentences)
                    sentence_count = 2
                else:
                    current_chunk = ""
                    sentence_count = 0
        
        # Final chunk
        if current_chunk.strip():
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": len(chunks),
                "chunk_type": "final",
                "sentence_count": sentence_count
            })
            
            chunks.append(DocumentChunk(
                content=current_chunk.strip(),
                metadata=chunk_metadata
            ))
        
        return chunks
    
    def _keyword_focused_chunks(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        """NEW: Create chunks focused on key insurance terms"""
        keywords = [
            "grace period", "waiting period", "premium", "coverage", "benefit",
            "exclusion", "claim", "policy", "deductible", "co-payment",
            "maternity", "pre-existing", "hospital", "treatment", "surgery"
        ]
        
        chunks = []
        words = text.split()
        
        for keyword in keywords:
            keyword_positions = []
            keyword_words = keyword.lower().split()
            
            # Find keyword positions
            for i in range(len(words) - len(keyword_words) + 1):
                if all(words[i + j].lower() == keyword_words[j] for j in range(len(keyword_words))):
                    keyword_positions.append(i)
            
            # Create chunks around keywords
            for pos in keyword_positions[:3]:  # Max 3 chunks per keyword
                start = max(0, pos - 100)
                end = min(len(words), pos + 200)
                
                chunk_text = " ".join(words[start:end])
                if len(chunk_text) > 300:  # Minimum chunk size
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk_id": len(chunks),
                        "chunk_type": "keyword_focused",
                        "focus_keyword": keyword
                    })
                    
                    chunks.append(DocumentChunk(
                        content=chunk_text,
                        metadata=chunk_metadata
                    ))
        
        return chunks[:10]  # Limit keyword chunks

# ═══════════════════════════════════════════════════════════════════════════════
#                           MULTI-LAYER SEARCH SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class MultiLayerSearchService:
    def __init__(self):
        self.embedding_model = EMBEDDING_MODEL
        self.faiss_index = FAISS_INDEX
        self.document_store = DOCUMENT_STORE
    
    def embed_documents(self, chunks: List[DocumentChunk]) -> None:
        """ENHANCED: Multi-strategy embedding"""
        if not chunks:
            return
        
        # Clear previous data for new document
        self.faiss_index.reset()
        self.document_store.clear()
        
        # Create embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
        
        # Store in FAISS
        self.faiss_index.add(embeddings.astype('float32'))
        
        # Store metadata
        for chunk in chunks:
            self.document_store.append({
                "content": chunk.content,
                "metadata": chunk.metadata
            })
        
        print(f"✅ Embedded {len(chunks)} chunks with multi-layer indexing")
    
    async def hybrid_search(self, query: str, top_k: int = 8) -> List[ClauseMatch]:
        """WINNING FEATURE: Multi-layer search"""
        if len(self.document_store) == 0:
            return []
        
        # Layer 1: Direct semantic search
        semantic_results = await self._semantic_search(query, top_k // 2)
        
        # Layer 2: Expanded query search (NEW)
        expanded_results = await self._expanded_query_search(query, top_k // 2)
        
        # Layer 3: Keyword boost search (NEW)
        keyword_results = await self._keyword_boost_search(query, top_k // 4)
        
        # Combine and deduplicate
        all_results = semantic_results + expanded_results + keyword_results
        return self._deduplicate_and_rank(all_results, top_k)
    
    async def _semantic_search(self, query: str, top_k: int) -> List[ClauseMatch]:
        """Enhanced semantic search"""
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        scores, indices = self.faiss_index.search(
            query_embedding.astype('float32'), min(top_k, len(self.document_store))
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.document_store) and idx >= 0:
                doc = self.document_store[idx]
                results.append(ClauseMatch(
                    content=doc["content"],
                    similarity_score=float(score),
                    source_document=doc["metadata"].get("source", "unknown"),
                    chunk_type="semantic"
                ))
        
        return results
    
    async def _expanded_query_search(self, query: str, top_k: int) -> List[ClauseMatch]:
        """NEW: Search with expanded query terms"""
        # Add insurance-specific terms
        expanded_query = f"{query} policy coverage benefit condition waiting period"
        
        query_embedding = self.embedding_model.encode([expanded_query], normalize_embeddings=True)
        scores, indices = self.faiss_index.search(
            query_embedding.astype('float32'), min(top_k, len(self.document_store))
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.document_store) and idx >= 0:
                doc = self.document_store[idx]
                results.append(ClauseMatch(
                    content=doc["content"],
                    similarity_score=float(score) * 0.9,  # Slight penalty for expanded
                    source_document=doc["metadata"].get("source", "unknown"),
                    chunk_type="expanded"
                ))
        
        return results
    
    async def _keyword_boost_search(self, query: str, top_k: int) -> List[ClauseMatch]:
        """NEW: Boost results that contain exact query keywords"""
        query_words = set(query.lower().split())
        boosted_results = []
        
        for i, doc in enumerate(self.document_store):
            content_words = set(doc["content"].lower().split())
            overlap = len(query_words.intersection(content_words))
            
            if overlap > 0:
                # Calculate boost score based on keyword overlap
                boost_score = overlap / len(query_words)
                
                boosted_results.append(ClauseMatch(
                    content=doc["content"],
                    similarity_score=boost_score,
                    source_document=doc["metadata"].get("source", "unknown"),
                    chunk_type="keyword_boost"
                ))
        
        # Sort by boost score and return top results
        boosted_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return boosted_results[:top_k]
    
    def _deduplicate_and_rank(self, results: List[ClauseMatch], top_k: int) -> List[ClauseMatch]:
        """Remove duplicates and rank by combined score"""
        seen_content = set()
        unique_results = []
        
        for result in sorted(results, key=lambda x: x.similarity_score, reverse=True):
            content_hash = hash(result.content[:200])  # Use first 200 chars as hash
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        return unique_results[:top_k]

# ═══════════════════════════════════════════════════════════════════════════════
#                              ENHANCED LLM SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancedLLMService:
    def __init__(self):
        self.model = LLM_MODEL
        self.generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=150,  # Keep responses short and concise
            temperature=0.05,
        )
    
    def create_concise_prompt(self, query: str, relevant_chunks: List[ClauseMatch]) -> str:
        """Create prompt optimized for concise, direct answers"""
        
        # Use top 3 most relevant chunks only
        top_chunks = sorted(relevant_chunks, key=lambda x: x.similarity_score, reverse=True)[:3]
        
        context = "\n\n".join([
            f"Document Section {i+1}:\n{chunk.content}"
            for i, chunk in enumerate(top_chunks)
        ])
        
        return f"""Answer this insurance policy question directly and concisely using the document context.
        Try to include more facts and numbers from the whole context provided

DOCUMENT CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Provide a clear, factual answer in 1-2 sentences
- Include specific numbers, periods, amounts, or conditions from the document
- Be direct and to the point but frame a good answer by considering the provided context only
- Avoid unnecessary explanations or generalizations
- Do not include section references or explanations

CONCISE ANSWER:"""
    
    def generate_answer(self, query: str, relevant_chunks: List[ClauseMatch]) -> str:
        """Generate concise answer without hardcoded patterns"""
        if not relevant_chunks:
            return "No relevant information found in the document for this query."
        
        prompt = self.create_concise_prompt(query, relevant_chunks)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config
                )
                
                if response.candidates and response.candidates[0].content.parts:
                    answer = response.text.strip()
                    
                    # Simple cleanup without pattern matching
                    answer = answer.replace('\n', ' ')  # Remove line breaks
                    answer = ' '.join(answer.split())   # Remove extra spaces
                    
                    return answer
                else:
                    return "Unable to generate response due to content filtering."
            
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + 1
                    print(f"Rate limit hit, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    return f"Error generating answer: {str(e)}"
        
        return "Failed to generate answer after multiple retries."
    
    async def batch_process_queries(self, queries: List[str], search_service: MultiLayerSearchService) -> List[str]:
        """Process multiple queries efficiently"""
        answers = []
        
        for i, query in enumerate(queries, 1):
            print(f"🔍 Processing Q{i}/{len(queries)}: {query[:60]}...")
            
            # Multi-layer search
            relevant_chunks = await search_service.hybrid_search(query, top_k=6)
            
            if relevant_chunks:
                best_score = max(chunk.similarity_score for chunk in relevant_chunks)
                print(f"📊 Found {len(relevant_chunks)} chunks (best score: {best_score:.3f})")
            else:
                print("⚠️ No relevant chunks found")
            
            # Generate concise answer
            answer = self.generate_answer(query, relevant_chunks)
            answers.append(answer)
            
            # Brief pause to avoid rate limits
            if i < len(queries):
                await asyncio.sleep(0.3)
        
        return answers


# ═══════════════════════════════════════════════════════════════════════════════
#                                   API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

# Initialize services
doc_processor = EnhancedDocumentProcessor()
search_service = MultiLayerSearchService()
llm_service = EnhancedLLMService()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != HACKRX_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"🚀 {request.method} {request.url.path} completed in {process_time:.2f}s")
    return response

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_submission(request: QueryRequest, token: str = Depends(verify_token)):
    """Enhanced submission endpoint with winning features"""
    try:
        print(f"🎯 Enhanced RAG Processing: {len(request.questions)} questions")
        print(f"📄 Document URL: {request.documents}")
        
        # Step 1: Download and identify document
        document_content = doc_processor.download_document(request.documents)
        file_type = doc_processor.get_file_type(request.documents, document_content)
        
        # Step 2: Extract text based on file type
        if file_type == 'pdf':
            text = doc_processor.extract_text_from_pdf(document_content)
        elif file_type == 'docx':
            text = doc_processor.extract_text_from_docx(document_content)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported document format: {file_type}")
        
        print(f"✅ Text extracted: {len(text)} characters")
        
        # Step 3: Intelligent multi-strategy chunking
        metadata = {"source": request.documents, "document_type": "policy"}
        chunks = doc_processor.intelligent_chunk_text(text, metadata)
        print(f"🧠 Created {len(chunks)} intelligent chunks using multiple strategies")
        
        # Step 4: Multi-layer embedding and indexing
        search_service.embed_documents(chunks)
        
        # Step 5: Enhanced batch query processing
        print(f"🔍 Starting multi-layer search for {len(request.questions)} questions...")
        answers = await llm_service.batch_process_queries(request.questions, search_service)
        
        print(f"✅ Enhanced RAG completed: {len(answers)} answers generated")
        return QueryResponse(answers=answers)
        
    except Exception as e:
        print(f"❌ Enhanced RAG Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "system": "Enhanced HackRx RAG v2.0",
        "features": [
            "Multi-layer search (semantic + expanded + keyword)",
            "Intelligent chunking (section + sliding + keyword-focused)",
            "Advanced LLM prompting",
            "Gemini 1.5-Flash integration",
            "FAISS vector search"
        ],
        "model": "gemini-1.5-flash",
        "embedding_model": "all-MiniLM-L6-v2",
        "indexed_chunks": len(DOCUMENT_STORE),
        "gemini_configured": bool(GEMINI_API_KEY)
    }

@app.get("/")
async def root():
    return {
        "message": "Enhanced HackRx LLM-Powered Query Retrieval System",
        "version": "2.0.0",
        "description": "Advanced RAG with multi-layer search and intelligent document processing",
        "endpoints": {
            "submit": "POST /hackrx/run",
            "health": "GET /health"
        }
    }

# ═══════════════════════════════════════════════════════════════════════════════
#                                   MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Enhanced HackRx RAG System...")
    print("🔧 Features: Multi-layer search, intelligent chunking, advanced prompting")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

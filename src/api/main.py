from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import time
from typing import List
from dotenv import load_dotenv
from urllib.parse import urlparse
from src.models.schemas import QueryRequest, QueryResponse
from src.services.document_processor import DocumentProcessor
from src.services.embedding_service import EmbeddingService
from src.services.llm_service import LLMService

# Load environment variables
load_dotenv()

app = FastAPI(title="LLM-Powered Query Retrieval System with Gemini")
security = HTTPBearer()

# Initialize services
doc_processor = DocumentProcessor()
embedding_service = EmbeddingService()
llm_service = LLMService(api_key=os.getenv("GEMINI_API_KEY"))

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    expected_token = os.getenv("HACKRX_TEAM_TOKEN", "1290a3dbfec356a3811cdfa5324f530e1e8a30b589b27adf6120bc227f54facd")
    if credentials.credentials != expected_token:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"Request to {request.url.path} took {process_time:.2f} seconds")
    return response

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_submission(request: QueryRequest, token: str = Depends(verify_token)):
    try:
        print(f"📄 Processing document: {request.documents}")
        print(f"❓ Number of questions: {len(request.questions)}")
        
        # Step 1: Download and process document
        document_content = doc_processor.download_document(request.documents)
        print("✅ Document downloaded successfully")
        
        # Step 2: Determine file type using improved method
        file_type = doc_processor.get_file_type(request.documents, document_content)
        
        # Step 3: Extract text based on file type
        if file_type == 'pdf':
            text = doc_processor.extract_text_from_pdf(document_content)
        elif file_type == 'docx':
            text = doc_processor.extract_text_from_docx(document_content)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported document format: {file_type}")
        
        print(f"✅ Text extracted successfully ({len(text)} characters)")
        
        # DEBUG: Print first 500 characters to verify content
        print(f"📝 First 500 chars: {text[:500]}...")
        
        # Step 4: Chunk the document
        metadata = {"source": request.documents, "document_type": "policy"}
        chunks = doc_processor.chunk_text(text, metadata)
        print(f"✅ Document chunked into {len(chunks)} pieces")
        
        # Step 5: Generate embeddings and store
        embedding_service.embed_documents(chunks)
        
        # Step 6: Process all queries with debug info
        answers = []
        for i, question in enumerate(request.questions, 1):
            print(f"\n🔍 Processing Q{i}: {question}")
            
            # Get relevant chunks
            relevant_chunks = embedding_service.search_similar_chunks(question, top_k=8)
            print(f"📊 Found {len(relevant_chunks)} relevant chunks")
            
            if relevant_chunks:
                best_score = max(chunk.similarity_score for chunk in relevant_chunks)
                print(f"🎯 Best similarity score: {best_score:.3f}")
            
            # Generate answer
            answer = llm_service.generate_answer(question, relevant_chunks)
            answers.append(answer)
        
        print(f"✅ All {len(answers)} questions processed successfully")
        
        return QueryResponse(answers=answers)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "message": "LLM Query Retrieval System with Gemini is running",
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
        "token_configured": bool(os.getenv("HACKRX_TEAM_TOKEN"))
    }

@app.get("/")
async def root():
    return {"message": "HackRx LLM-Powered Query Retrieval System", "status": "active"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

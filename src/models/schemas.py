from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    documents: str  # Blob URL
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class DocumentChunk(BaseModel):
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class ClauseMatch(BaseModel):
    content: str
    similarity_score: float
    source_document: str
    page_number: Optional[int] = None

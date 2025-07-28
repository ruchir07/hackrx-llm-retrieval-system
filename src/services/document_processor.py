import requests
import PyPDF2
from docx import Document
from typing import List, Dict
import io
from urllib.parse import urlparse
from src.models.schemas import DocumentChunk

class DocumentProcessor:
    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 200
    
    def get_file_type(self, url: str, content: bytes = None) -> str:
        """Determine file type from URL or content"""
        # First try to get from URL path
        parsed_url = urlparse(url)
        file_path = parsed_url.path.lower()
        
        if file_path.endswith('.pdf'):
            return 'pdf'
        elif file_path.endswith('.docx'):
            return 'docx'
        
        # If URL doesn't help, try content-based detection
        if content:
            if content.startswith(b'%PDF'):
                return 'pdf'
            elif content.startswith(b'PK\x03\x04') and b'word/' in content:
                return 'docx'
        
        # Default fallback
        return 'unknown'
    
    def download_document(self, url: str) -> bytes:
        """Download document from blob URL"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            raise Exception(f"Failed to download document: {str(e)}")
    
    def extract_text_from_pdf(self, content: bytes) -> str:
        try:
            pdf_file = io.BytesIO(content)
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
        
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        text += f"\n[Page {page_num + 1}]\n{page_text}\n"
                except Exception as e:
                    print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
                    continue
        
        # Clean up the text
            text = text.replace('\n\n\n', '\n\n')  # Reduce excessive newlines
        
            print(f"📄 Extracted {len(text)} characters from {len(reader.pages)} pages")
            return text
        
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")

    
    def extract_text_from_docx(self, content: bytes) -> str:
        """Extract text from DOCX"""
        try:
            doc_file = io.BytesIO(content)
            doc = Document(doc_file)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            raise Exception(f"Failed to extract text from DOCX: {str(e)}")
    
    def chunk_text(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        """Split text into overlapping chunks optimized for insurance documents"""
        chunks = []
    
    # Split by sentences first, then group into chunks
        sentences = text.replace('\n', ' ').split('. ')
    
        current_chunk = ""
        sentence_count = 0
    
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
        # Add sentence to current chunk
            if current_chunk:
                current_chunk += ". " + sentence
            else:
                current_chunk = sentence
            
            sentence_count += 1
        
        # Create chunk when we have enough content
            if len(current_chunk) >= 800 or sentence_count >= 10:
                if current_chunk.strip():
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk_id": len(chunks),
                        "sentence_count": sentence_count,
                        "char_length": len(current_chunk)
                    })
                
                    chunks.append(DocumentChunk(
                        content=current_chunk.strip(),
                        metadata=chunk_metadata
                    ))
            
            # Start new chunk with overlap (last 2 sentences)
                sentences_in_chunk = current_chunk.split('. ')
                if len(sentences_in_chunk) >= 2:
                    current_chunk = '. '.join(sentences_in_chunk[-2:])
                    sentence_count = 2
                else:
                    current_chunk = ""
                    sentence_count = 0
    
    # Add final chunk if there's remaining content
        if current_chunk.strip():
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": len(chunks),
                "sentence_count": sentence_count,
                "char_length": len(current_chunk)
            })
        
            chunks.append(DocumentChunk(
                content=current_chunk.strip(),
                metadata=chunk_metadata
            ))
    
        print(f"📊 Created {len(chunks)} chunks (avg {sum(len(c.content) for c in chunks)//len(chunks)} chars each)")
        return chunks

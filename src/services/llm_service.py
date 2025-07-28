import google.generativeai as genai
from typing import List
import os
from src.models.schemas import ClauseMatch

class LLMService:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Configure generation parameters for better performance
        self.generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=400,
            temperature=0.05,
        )
    
    def create_optimized_prompt(self, query: str, relevant_chunks: List[ClauseMatch]) -> str:
        """Create better prompt with more context"""
    
    # Use top 4 chunks for better coverage
        top_chunks = sorted(relevant_chunks, key=lambda x: x.similarity_score, reverse=True)[:4]
    
        context = "\n\n---\n\n".join([
            f"Document Section {i+1} (Relevance: {chunk.similarity_score:.3f}):\n{chunk.content}"
            for i, chunk in enumerate(top_chunks)
        ])
    
        return f"""You are analyzing an insurance policy document to answer specific questions.

DOCUMENT CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Carefully read through ALL the document sections above
- Look for information that directly answers the question
- Include specific details like time periods, percentages, amounts, conditions
- If you find the answer, provide it clearly and completely
- Only say "Information not available" if you've thoroughly checked all sections and the answer is truly not there
- For insurance terms, be precise about waiting periods, coverage limits, and conditions

ANSWER:"""

    def search_similar_chunks(self, query: str, top_k: int = 8) -> List[ClauseMatch]:  # Increased from 5 to 8
        """Search for similar document chunks with better coverage"""
    
        if len(self.document_store) == 0:
            return []
    
        # Create expanded query for better matching
        expanded_query = f"{query} policy insurance coverage benefit waiting period condition"
    
        query_embedding = self.model.encode([expanded_query], normalize_embeddings=True)
    
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
    
    def generate_answer(self, query: str, relevant_chunks: List[ClauseMatch]) -> str:
        """Generate answer using Gemini with retrieved context"""
        
        if not relevant_chunks:
            return "No relevant information found in the document for this query."
        
        prompt = self.create_optimized_prompt(query, relevant_chunks)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            if response.candidates and response.candidates[0].content.parts:
                return response.text.strip()
            else:
                return "Unable to generate response due to content filtering."
        
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def batch_process_queries(self, queries: List[str], embedding_service) -> List[str]:
        """Process multiple queries efficiently"""
        answers = []
        
        for i, query in enumerate(queries, 1):
            print(f"Processing question {i}/{len(queries)}: {query[:50]}...")
            
            # Retrieve relevant chunks
            relevant_chunks = embedding_service.search_similar_chunks(query, top_k=5)
            
            # Generate answer
            answer = self.generate_answer(query, relevant_chunks)
            answers.append(answer)
        
        return answers

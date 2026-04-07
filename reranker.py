"""
LLM-based reranker for improving retrieval precision using Gemini.
"""

from typing import List, Optional
from langchain_core.documents import Document

class LLMReranker:
    """Uses Gemini LLM to rerank documents by relevance."""
    
    def __init__(self, model_name: str = "models/gemini-2.5-flash"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        self.model_name = model_name
    
    def rerank(self, query: str, docs: List[Document], top_k: Optional[int] = None) -> List[Document]:
        """Rerank documents using LLM evaluation."""
        if not docs:
            return docs
        
        # Create prompt for ranking
        doc_list = "\n\n".join([
            f"[{i}] {doc.page_content[:500]}..." 
            for i, doc in enumerate(docs, 1)
        ])
        
        prompt = f"""Rank these documents by relevance to the query. Return only indices as comma-separated list (e.g., "1,3,2,4").

Query: {query}

Documents:
{doc_list}

Ranking (indices only):"""
        
        response = self.llm.invoke(prompt)
        response_text = str(response.content).strip()
        try:
            # Parse the ranking
            indices = [int(x.strip()) - 1 for x in response_text.split(",")]
            reranked = [docs[i] for i in indices if 0 <= i < len(docs)]
            return reranked[:top_k] if top_k else reranked
        except (ValueError, IndexError):
            # If parsing fails, return original order
            return docs[:top_k] if top_k else docs

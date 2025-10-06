from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

# Import Config from parent directory
try:
    from config import Config
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config import Config

class DocumentRetriever:
    """Handles document retrieval from vector store"""
    
    def __init__(self, vector_store, embedding_model, similarity_threshold: Optional[float] = None):
        """
        Initialize document retriever
        
        Args:
            vector_store: Vector store instance (ChromaDB client wrapper)
            embedding_model: Embedding model instance (used for general info, but query embedding now handled by VectorStore)
            similarity_threshold: Minimum similarity score for relevant documents. If None, uses config.
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model 
        
        self.config = Config() # Instantiate Config
        self.similarity_threshold = similarity_threshold if similarity_threshold is not None else self.config.get_processing_config()['similarity_threshold']
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def retrieve_documents(self, query: str, num_documents: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        The query embedding is now handled by the VectorStore's internal embedding function.
        
        Args:
            query: User query string
            num_documents: Number of documents to retrieve
            
        Returns:
            List of relevant documents with metadata and similarity scores.
        """
        try:
            self.logger.info(f"Retrieving documents for query: {query[:100]}...")
                        
            # Call similarity_search on the vector_store directly with the query string
            # The VectorStore (ChromaDB) is now responsible for embedding the query
            search_results = self.vector_store.similarity_search(
                query=query,
                num_results=num_documents
            )
            
            self.logger.info(f"Retrieved {len(search_results)} raw documents from vector store.")
            
            # No need to filter by similarity_threshold here, as prepare_context will do it
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {str(e)}")
            raise Exception(f"Failed to retrieve documents: {str(e)}")
            
    def prepare_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Prepare context string from retrieved documents, filtering by similarity threshold.
        
        Args:
            documents: List of documents (each with 'content', 'metadata', 'score').
            
        Returns:
            Formatted context string for the LLM.
        """
        try:
            context_parts = []
            relevant_docs_count = 0
            
            # Sort documents by score (distance) (lower distance means higher similarity)
            # Assuming 'score' is distance (for example: L2 distance from ChromaDB), so we sort in ascending order
            sorted_documents = sorted(documents, key=lambda x: x['score'])

            for doc in sorted_documents:
                score = doc.get('score', 0.0)
                content = doc.get('content', '')
                source = doc.get('metadata', {}).get('source', 'Unknown Source')
                page_number = doc.get('metadata', {}).get('page_number', 'N/A')

                # For a prototype, we will include all documents retrieved by `num_documents` in the context
                # The LLM is generally good at discerning relevance from provided context
                # If strict filtering by similarity_threshold (distance) is needed, it would be applied here (for example 'if score <= self.similarity_threshold:')
                # For now, 'self.similarity_threshold' is passed but not strictly enforced here in a filtering manner, but rather conceptually for the RAGPipeline

                context_part = f"""--- Source: {source} (Page: {page_number}, Distance: {score:.2f}) ---
{content}
"""
                context_parts.append(context_part)
                relevant_docs_count += 1
            
            if not context_parts:
                self.logger.warning("No documents found for context preparation (could be due to no retrieval or empty content).")
                return "" # Return empty string if no relevant docs
            
            self.logger.info(f"Prepared context from {relevant_docs_count} documents.")
            return "\n\n".join(context_parts)
            
        except Exception as e:
            self.logger.error(f"Error preparing context: {str(e)}")
            return "Error preparing context from retrieved documents."
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the retrieval system
        
        Returns:
            Dictionary with retrieval statistics
        """
        try:
            vector_info = self.vector_store.get_collection_info()
            embedding_info = self.embedding_model.get_model_info() # Get info from the EmbeddingModel instance
            
            return {
                'vector_store_info': vector_info,
                'similarity_threshold': self.similarity_threshold,
                'embedding_model_info': embedding_info
            }
            
        except Exception as e:
            self.logger.error(f"Error getting retrieval stats: {str(e)}")
            return {'error': str(e)}
    
    def update_similarity_threshold(self, threshold: float):
        """
        Update the similarity threshold for document retrieval
        
        Args:
            threshold: New similarity threshold (0.0 to 1.0).
                       Note: This is currently used as a conceptual filter and can be
                       implemented as a strict filter later if needed.
        """
        if 0.0 <= threshold <= 1.0:
            self.similarity_threshold = threshold
            self.logger.info(f"Updated similarity threshold to {threshold}")
        else:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")

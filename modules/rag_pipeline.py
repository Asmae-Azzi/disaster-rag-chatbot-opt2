from typing import Dict, Any, List
import logging
from pathlib import Path

# Import Config from parent directory
try:
    from config import Config
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config import Config

from modules.retriever import DocumentRetriever
from modules.llm_handler import LLMHandler

class RAGPipeline:
    """Complete RAG (Retrieval Augmented Generation) pipeline"""
    
    def __init__(self, vector_store, metadata_store, embedding_model, config: Config): # Accept config object
        """
        Initialize RAG pipeline
        
        Args:
            vector_store: Vector store instance
            metadata_store: Metadata store instance
            embedding_model: Embedding model instance
            config: Config instance for application settings
        """
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.embedding_model = embedding_model
        self.config = config # Store the config object
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components using config settings
        try:
            processing_config = self.config.get_processing_config()
            retrieval_config = self.config.get_retrieval_config()
            llm_config = self.config.get_llm_config()

            self.retriever = DocumentRetriever(
                vector_store=vector_store,
                embedding_model=embedding_model,
                similarity_threshold=processing_config['similarity_threshold'] # Get from config
            )
            
            self.llm_handler = LLMHandler(
                model_type=llm_config['model_type'],
                model_name=llm_config['model_name'],
                api_key=llm_config['openai_api_key'], 
                api_base=llm_config['openrouter_api_base'],
                max_tokens=llm_config['max_tokens'] # Pass max_tokens to LLMHandler
            )
            
            # Store default retrieval settings from config
            self.default_num_documents = retrieval_config['default_num_documents']

            self.logger.info("RAG pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing RAG pipeline: {str(e)}")
            raise Exception(f"Failed to initialize RAG pipeline: {str(e)}")

    def query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline.
        
        Args:
            query: User's question.
            
        Returns:
            Dictionary containing the response, sources, and other information.
        """
        try:
            self.logger.info(f"Processing query: {query[:100]}...")
            
            # 1. Retrieve relevant documents
            # Use default_num_documents from config for retrieval
            retrieved_documents = self.retriever.retrieve_documents(query, num_documents=self.default_num_documents)
            
            # Prepare context string for LLM (this also filters by similarity_threshold internally)
            context_for_llm = self.retriever.prepare_context(retrieved_documents) # Use the public method
            
            if not context_for_llm: # If no relevant documents passed the threshold or were retrieved
                return {
                    'response': "I couldn't find relevant information for your query. Please try rephrasing your question or ask about a different topic.",
                    'sources': [],
                    'num_documents_retrieved': 0,
                    'no_relevant_docs': True,
                    'error': False
                }
            
            # Extract sources for display (from all retrieved, not just those used in context)
            sources_list = list(set([doc['metadata'].get('source', 'Unknown Source') for doc in retrieved_documents if 'metadata' in doc and 'source' in doc['metadata']]))
            
            # 2. Generate response using LLM
            # Pass relevant_content for simple LLM, and context_for_llm for LLM
            llm_response_data = self.llm_handler.generate_response(
                prompt=query, 
                context=context_for_llm, 
                relevant_content=[doc['content'] for doc in retrieved_documents] # Pass raw content for simple mode
            )
            
            return {
                'response': llm_response_data['response'],
                'sources': sources_list,
                'model_used': llm_response_data['model_used'],
                'tokens_used': llm_response_data['tokens_used'],
                'num_documents_retrieved': len(retrieved_documents),
                'error': llm_response_data['error']
            }
        except Exception as e:
            self.logger.error(f"RAG pipeline query failed: {str(e)}")
            return {
                'response': f"An error occurred while generating the response: {str(e)}",
                'sources': [],
                'num_documents_retrieved': 0,
                'error': True
            }
    
    def update_retrieval_settings(self, similarity_threshold: float = None, num_documents: int = None):
        """
        Update retrieval settings.
        
        Args:
            similarity_threshold: New similarity threshold (0.0 to 1.0). If None, keeps current.
            num_documents: New default number of documents to retrieve. If None, keeps current.
        """
        try:
            if similarity_threshold is not None:
                self.retriever.update_similarity_threshold(similarity_threshold)
                # Update the config object directly to reflect changes in sidebar immediately
                self.config.SIMILARITY_THRESHOLD = similarity_threshold 
            
            if num_documents is not None:
                self.default_num_documents = num_documents 
                # Update the config object directly to reflect changes in sidebar immediately
                self.config.DEFAULT_NUM_DOCUMENTS = num_documents 
            
            self.logger.info("Retrieval settings updated successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating retrieval settings: {str(e)}")
            raise Exception(f"Failed to update settings: {str(e)}")

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the RAG pipeline components"""
        try:
            llm_info = self.llm_handler.get_model_info()
            retriever_info = self.retriever.get_retrieval_stats()
            
            return {
                'llm_info': llm_info,
                'retriever_info': retriever_info,
                'pipeline_status': 'Initialized'
            }
        except Exception as e:
            self.logger.error(f"Error getting pipeline info: {str(e)}")
            return {'error': str(e)}
    
    def test_pipeline(self) -> Dict[str, Any]:
        """Test the RAG pipeline with a simple query"""
        try:
            test_query = "What should I do to prepare for an emergency?"
            result = self.query(test_query) # Use default num_documents
            
            return {
                'test_successful': not result['error'],
                'test_query': test_query,
                'response_length': len(result['response']),
                'documents_retrieved': result['num_documents_retrieved'],
                'sources_found': len(result['sources'])
            }
            
        except Exception as e:
            return {
                'test_successful': False,
                'error': str(e)
            }


import numpy as np
from typing import List, Union, Dict, Any, Optional
import logging
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Import Config from parent directory
try:
    from config import Config
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config import Config

class EmbeddingModel:
    """Handles text embeddings using a Sentence-Transformers model"""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedding model with a Sentence-Transformers model.
        
        Args:
            model_name: Name of the Sentence-Transformers model to use.
                        Defaults to value from Config if None.
        """
        self.config = Config() # Instantiate Config
        self.model_name = model_name if model_name is not None else self.config.get_embedding_config()['model_name']
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        try:
            self.logger.info(f"Initializing Sentence-Transformers model: {self.model_name}")
            # Load the pre-trained Sentence-Transformers model
            # Explicitly set device to 'cpu' for compatibility with Streamlit Cloud's free tier and to avoid issues if GPU is not configured.
            self.device = 'cpu'
            self.model = SentenceTransformer(self.model_name, device=self.device) 
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            
            self.logger.info(f"Sentence-Transformers model initialized successfully. Embedding dimension: {self.embedding_dimension}")
        except Exception as e:
            self.logger.error(f"Failed to load Sentence-Transformers model '{self.model_name}': {str(e)}")
            raise Exception(f"Failed to initialize embedding model: {str(e)}")

    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text (or a list of texts) into embeddings.
        
        Args:
            text: Single string or list of strings to encode.
            
        Returns:
            NumPy array of embeddings.
        """
        try:
            # SentenceTransformer handles single string or list of strings
            embeddings = self.model.encode(text, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            self.logger.error(f"Error encoding text: {str(e)}")
            raise Exception(f"Failed to encode text: {str(e)}")

    def get_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between -1 and 1 (often normalized to 0-1)
        """
        try:
            embedding1 = self.encode_text(text1)
            embedding2 = self.encode_text(text2)
            
            # Ensure cosine similarity is correct even if embeddings are not normalized
            similarity = float(np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}")
            raise Exception(f"Failed to calculate similarity: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information.
        """
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'model_type': 'Sentence-Transformers',
            'max_sequence_length': self.model.max_seq_length, # Get actual max sequence length
            'device': self.device
        }


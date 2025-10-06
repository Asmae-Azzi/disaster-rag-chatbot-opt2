import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import uuid
from pathlib import Path
import logging

# Import Config from the parent directory
try:
    from config import Config
except ImportError:
    # Fallback for direct testing if modules is not on path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """ChromaDB-based vector store for document embeddings"""

    def __init__(self, embedding_model, collection_name: str = "disaster_preparedness"):
        """
        Initialize ChromaDB vector store.
        Can connect to a local persistent client or an external HTTP client.
        """
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        config = Config() # Instantiate Config to get settings

        chroma_host = config.CHROMA_DB_HOST
        chroma_port = config.CHROMA_DB_PORT
        persist_directory = Path(config.CHROMA_DB_PATH)

        # ChromaDB requires a specific way to pass the embedding function if using external models
        # For Sentence-Transformers, I use Chroma's built-in function or directly pass embeddings
        # For simplicity with `add_documents` expecting `embeddings`, I'll keep `None` here and generate outside
        # For `similarity_search` in `DocumentRetriever`, it needs an embedding function if `query` is passed
        # I use the SentenceTransformerEmbeddingFunction from chromadb.utils for consistency
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model.model_name,
            device=embedding_model.device # Ensure device matches what's used in EmbeddingModel
        )

        if chroma_host:
            # Use HttpClient for external ChromaDB server
            logger.info(f"Connecting to external ChromaDB at {chroma_host}:{chroma_port}")
            self.client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True # Allow resetting collection
                )
            )
        else:
            # Use PersistentClient for local ChromaDB
            logger.info(f"Using local persistent ChromaDB at {persist_directory}")
            persist_directory.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            self.client = chromadb.PersistentClient(
                path=str(persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True # Allow resetting collection
                )
            )

        # Create or get collection. Pass the embedding function directly here
        # If the collection already exists and has a different embedding function, this might cause issues
        # For existing collections, you retrieve without specifying embedding_function unless recreating
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function # Pass it here for retrieval
            )
            logger.info(f"Retrieved existing ChromaDB collection: {self.collection_name}")
        except Exception as e:
            logger.info(f"Collection '{self.collection_name}' not found or error retrieving, creating new one.")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function, # Pass it here for creation
                metadata={"description": "Disaster preparedness documents"}
            )
            logger.info(f"Created new ChromaDB collection: {self.collection_name}")


    def add_documents(self, chunks: List[Dict[str, Any]], source_filename: str):
        """Add document chunks to the vector store"""
        try:
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            embeddings = [] # I will generate embeddings once and pass them

            for i, chunk in enumerate(chunks):
                # Generate embedding for the chunk
                chunk_embedding = self.embedding_model.encode_text(chunk['content']).tolist() # Convert to list

                documents.append(chunk['content'])
                metadatas.append({
                    "source": source_filename,
                    "chunk_index": chunk['chunk_index'],
                    "page_number": chunk.get('page_number', 0), # Ensure page_number is included
                    "chunk_size": chunk.get('chunk_size', len(chunk['content']))
                })
                # Using a unique ID for each chunk
                ids.append(f"{source_filename}-{chunk['chunk_index']}-{uuid.uuid4().hex[:8]}")
                embeddings.append(chunk_embedding)

            # Add to ChromaDB in batches
            batch_size = 500 # Adjust batch size if needed
            for i in range(0, len(documents), batch_size):
                batch_documents = documents[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]

                self.collection.add(
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    embeddings=batch_embeddings, # Pass pre-generated embeddings
                    ids=batch_ids
                )
            logger.info(f"Added {len(documents)} chunks from {source_filename} to ChromaDB.")

        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            raise Exception(f"Failed to add documents to vector store: {e}")

    def similarity_search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform similarity search in the vector store using the embedding model.
        Now uses ChromaDB's built-in embedding function for queries.
        """
        try:
            # ChromaDB's client.query uses its configured embedding_function if query text is provided
            results = self.collection.query(
                query_texts=[query],
                n_results=num_results,
                include=['documents', 'metadatas', 'distances']
            )

            # Format results
            formatted_results = []
            if results and results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'score': results['distances'][0][i]
                    })
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            raise Exception(f"Failed to perform similarity search: {str(e)}")

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the ChromaDB collection"""
        try:
            count = self.collection.count()
            # Note: persist_directory might not be meaningful for HttpClient
            # I'll return it if it's a PersistentClient, otherwise indicate N/A
            chroma_host = Config().CHROMA_DB_HOST # Re-read config for display
            
            info = {
                'collection_name': self.collection_name,
                'document_count': count,
            }
            if not chroma_host: # Only show persist_directory if local
                info['persist_directory'] = str(Config().CHROMA_DB_PATH)
            else:
                info['chroma_server_host'] = chroma_host
                info['chroma_server_port'] = Config().CHROMA_DB_PORT
            
            return info
        except Exception as e:
            raise Exception(f"Failed to get collection info: {str(e)}")

    def delete_documents_by_source(self, source_filename: str):
        """Delete all documents from a specific source"""
        try:
            # Get all documents with the specified source
            results = self.collection.get(
                where={"source": source_filename},
                include=['metadatas']
            )

            if results['ids']:
                # Delete the documents
                self.collection.delete(ids=results['ids'])
                return len(results['ids'])

            return 0

        except Exception as e:
            raise Exception(f"Failed to delete documents: {str(e)}")

    def reset_collection(self):
        """Reset the entire collection (delete all documents)"""
        try:
            # For HttpClient, this might need a different approach if client.delete_collection is not supported but usually it is
            self.client.delete_collection(self.collection_name)

            # Recreate the collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Disaster preparedness documents"}
            )
            return True

        except Exception as e:
            raise Exception(f"Failed to reset collection: {str(e)}")

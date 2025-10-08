import os
import sys
import re
import logging
import boto3
from pathlib import Path
from typing import List, Optional
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- FIX for ModuleNotFoundError ---
# This line adds the project's root directory to the system path,
# ensuring all local modules can be found.
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
# --- End of FIX ---

# Load environment variables FIRST
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
    logger.info("Environment variables loaded successfully")
except ImportError as e:
    logger.error(f"Failed to load dotenv: {e}")
    logger.warning("Continuing without .env file")
except Exception as e:
    logger.error(f"Unexpected error loading environment: {e}")
    sys.exit(1)

# Now import local modules after environment is loaded
try:
    from langchain.text_splitter import CharacterTextSplitter
    from modules.vector_store import VectorStore
    from modules.pdf_processor import PDFProcessor
    from modules.embeddings import EmbeddingModel
    from modules.metadata_store import MetadataStore
    logger.info("All required modules imported successfully")
except ImportError as e:
    logger.error(f"Failed to import required module: {e}")
    logger.error("Please ensure all dependencies are installed and modules exist")
    sys.exit(1)

# --- Python sys.path for debugging ---
logger.debug("--- Python sys.path for debugging ---")
for p in sys.path:
    logger.debug(p)
logger.debug("--- End sys.path debugging ---")

# Define variables with validation (could be moved to Config if preferred)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def load_documents_from_s3(pdf_processor):  #"Load documents directly from S3 bucket"
    try:
        # Get S3 credentials from environment
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )

        bucket_name = os.getenv('S3_BUCKET_NAME', 'your-bucket-name')
        prefix = os.getenv('S3_PREFIX', 'documents/')

        logger.info(f"Connecting to S3 bucket: {bucket_name}")
        logger.info(f"Using S3 prefix: '{prefix}'")

        def _load_with_prefix(current_prefix: str):
            loaded_docs = []
            paginator = s3_client.get_paginator('list_objects_v2')
            page_iter = paginator.paginate(Bucket=bucket_name, Prefix=current_prefix)

            total_objects = 0
            for page in page_iter:
                for obj in page.get('Contents', []) or []:
                    total_objects += 1
                    key = obj['Key']
                    if key.lower().endswith('.pdf'):
                        logger.info(f"Processing S3 object: {key}")
                        try:
                            pdf_response = s3_client.get_object(Bucket=bucket_name, Key=key)
                            pdf_content = pdf_response['Body'].read()
                            chunks = pdf_processor.process_pdf(pdf_content, filename=key)
                            loaded_docs.extend(chunks)
                            logger.info(f"Successfully processed {key}: {len(chunks)} chunks")
                        except Exception as e:
                            logger.error(f"Error processing {key}: {e}")
                            continue
            logger.info(f"Objects scanned under prefix '{current_prefix}': {total_objects}")
            return loaded_docs

        # First try with the provided prefix
        raw_documents = _load_with_prefix(prefix)

        # If nothing found, retry without prefix as a fallback
        if not raw_documents:
            logger.warning(f"No PDF documents found under prefix '{prefix}'. Retrying with no prefix...")
            raw_documents = _load_with_prefix('')

        logger.info(f"Total documents loaded from S3: {len(raw_documents)}")
        return raw_documents

    except Exception as e:
        logger.error(f"Error connecting to S3: {e}")
        return []

# Validate configuration
def validate_config() -> bool:
    """Validate all configuration parameters."""
    try:
        # Validate chunk parameters
        if CHUNK_SIZE <= 0:
            logger.error("CHUNK_SIZE must be positive")
            return False
        if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
            logger.error("CHUNK_OVERLAP must be non-negative and less than CHUNK_SIZE")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

def clean_text(text: str) -> str:
    """Cleans and preprocesses the given text."""
    try:
        if not isinstance(text, str):
            logger.warning(f"Expected string, got {type(text)}")
            text = str(text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        logger.error(f"Text cleaning failed: {e}")
        return str(text) if text else ""

def create_chunks(raw_docs: List) -> Optional[List]:
    """Splits the loaded documents into chunks when needed.
    If input already consists of chunk dicts from PDFProcessor, return as-is to preserve metadata.
    """
    try:
        if not raw_docs:
            logger.warning("No documents provided for chunking")
            return []
        
        logger.info(f"Loaded {len(raw_docs)} documents.")
        
        # If already chunk dicts produced by PDFProcessor, skip re-splitting to preserve 'source'
        if raw_docs and isinstance(raw_docs[0], dict) and 'content' in raw_docs[0] and 'chunk_index' in raw_docs[0]:
            logger.info("Input appears already chunked by PDFProcessor; skipping additional splitting.")
            return raw_docs
        
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        
        docs = text_splitter.split_documents(raw_docs)
        logger.info(f"Created {len(docs)} chunks.")
        return docs
        
    except Exception as e:
        logger.error(f"Failed to create chunks: {e}")
        return None

def create_vector_store(docs: List, embeddings) -> bool:
    """Creates and saves a vector store from the document chunks."""
    try:
        if not docs:
            logger.error("No documents provided for vector store creation")
            return False
        
        # Fix: VectorStore only takes embedding_model and optional collection_name
        vector_store = VectorStore(embedding_model=embeddings)
        
        # Add documents to the vector store
        for i, doc in enumerate(docs):
            # Support either LangChain Document or dict chunk
            if isinstance(doc, dict):
                source_filename = doc.get('source', 'unknown')
                doc_dict = {
                    'content': doc.get('content', ''),
                    'metadata': doc.get('metadata', {}),
                    'chunk_index': doc.get('chunk_index', i),
                    'page_number': doc.get('page_number', 0),
                    'chunk_size': doc.get('chunk_size', len(doc.get('content', '')))
                }
            else:
                # LangChain Document
                meta = getattr(doc, 'metadata', {}) or {}
                source_filename = meta.get('filename', 'unknown')
                doc_dict = {
                    'content': getattr(doc, 'page_content', ''),
                    'metadata': meta,
                    'chunk_index': i
                }
            try:
                vector_store.add_documents([doc_dict], source_filename)
            except Exception as e:
                logger.error(f"Failed to add document {i}: {e}")
                continue
        
        logger.info("Vector store created and documents added successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        return False

def update_metadata_store(docs: List) -> bool:
    """Aggregate and write per-document metadata (counts and approximate size)."""
    try:
        store = MetadataStore()
        per_source = {}
        for doc in docs:
            if isinstance(doc, dict):
                source = doc.get('source', 'unknown')
                size = int(doc.get('chunk_size', len(doc.get('content', ''))))
            else:
                meta = getattr(doc, 'metadata', {}) or {}
                source = meta.get('filename', 'unknown')
                size = len(getattr(doc, 'page_content', '') or '')
            if source not in per_source:
                per_source[source] = {'num_chunks': 0, 'total_size': 0}
            per_source[source]['num_chunks'] += 1
            per_source[source]['total_size'] += size
        for source, stats in per_source.items():
            store.add_document(filename=source, file_size=stats['total_size'], num_chunks=stats['num_chunks'])
        logger.info("Metadata store updated for %d sources.", len(per_source))
        return True
    except Exception as e:
        logger.error(f"Failed to update metadata store: {e}")
        return False

def main() -> bool:
    """Main execution function with comprehensive error handling."""
    print("DEBUG: CHROMA_DB_PATH =", Config().CHROMA_DB_PATH)

    try:
        logger.info("Starting database setup...")
        
        # Validate configuration
        if not validate_config():
            logger.error("Configuration validation failed")
            return False
        
        # Initialize components
        logger.info("Initializing document loader...")
        pdf_processor = PDFProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        
        # Load documents from S3 - pass the pdf_processor
        logger.info("Loading documents from S3...")
        raw_documents = load_documents_from_s3(pdf_processor)
        
        if not raw_documents:
            logger.warning("No documents were loaded from S3. Continuing without documents so the app can start. Verify S3 settings and permissions.")
            return True
        
        logger.info("Initializing embeddings model...")
        embeddings_model = EmbeddingModel(model_name='all-MiniLM-L6-v2')
        if not embeddings_model:
            logger.error("Failed to initialize embeddings model")
            return False
        
        # Pre-process documents (if needed)
        logger.info("Preparing document chunks...")
        processed_documents = create_chunks(raw_documents)
        if processed_documents is None:
            logger.error("Document chunking failed")
            return False
        
        # Update metadata store
        logger.info("Updating metadata store...")
        update_metadata_store(processed_documents)
        
        # Create and save the vector store
        logger.info("Creating vector store...")
        if not create_vector_store(processed_documents, embeddings_model):
            logger.error("Vector store creation failed")
            return False
        
        logger.info("Database setup completed successfully!")
        return True
        
    except KeyboardInterrupt:
        logger.info("Database setup interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during database setup: {e}")
        return False

if __name__ == '__main__':
    success = main()
    if not success:
        logger.error("Database setup failed")
        sys.exit(1)
    else:
        logger.info("Database setup completed successfully")
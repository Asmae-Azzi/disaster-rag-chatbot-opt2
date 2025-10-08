import os


class Config:
    def __init__(self):
        # Retrieve keys from environment variables
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        # Unify region: prefer AWS_REGION; fallback to AWS_REGION_NAME
        self.aws_region_name = os.environ.get("AWS_REGION") or os.environ.get("AWS_REGION_NAME")
        self.s3_bucket_name = os.environ.get("S3_BUCKET_NAME")
    
    # ChromaDB settings
    CHROMA_DB_HOST = None  # None for local, or "localhost" for external
    CHROMA_DB_PORT = 8000
    CHROMA_DB_PATH = "./chroma_db"
    
    # Embedding settings
    def get_embedding_config(self):
        return {
            'model_name': 'all-MiniLM-L6-v2'
        }
        
    # Processing settings
    def get_processing_config(self):
        return {
            'chunk_size': 500,
            'chunk_overlap': 50,
            'similarity_threshold': 0.7
        }
    
    # Retrieval settings
    def get_retrieval_config(self):
        return {
            'default_num_documents': 5
        }
    
    # LLM settings
    def get_llm_config(self):
        return {
            'model_type': 'openai',
            'model_name': 'gpt-3.5-turbo',
            'openai_api_key': self.openai_api_key,
            'openrouter_api_base': 'https://openrouter.ai/api/v1',
            'max_tokens': 1000
        }
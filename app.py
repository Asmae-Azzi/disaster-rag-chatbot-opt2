import streamlit as st
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our custom modules with error handling
try:
    from modules.vector_store import VectorStore
    from modules.metadata_store import MetadataStore
    from modules.pdf_processor import PDFProcessor
    from modules.embeddings import EmbeddingModel
    from modules.rag_pipeline import RAGPipeline
    logger.info("All required modules imported successfully")
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.error("Please ensure all modules exist and dependencies are installed")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="Disaster Preparedness Chatbot",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state with validation
def init_session_state():
    """Initialize session state with proper validation"""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'metadata_store' not in st.session_state:
        st.session_state.metadata_store = None
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    if 'initialization_error' not in st.session_state:
        st.session_state.initialization_error = None

init_session_state()

@st.cache_resource
def get_embedding_model_cached():
    return EmbeddingModel(model_name='all-MiniLM-L6-v2')

@st.cache_resource
def get_vector_store_cached():
    embedding_model = get_embedding_model_cached()
    return VectorStore(embedding_model=embedding_model)

@st.cache_resource
def get_metadata_store_cached():
    return MetadataStore()

@st.cache_resource
def get_rag_pipeline_cached():
    from config import Config
    config = Config()
    return RAGPipeline(
        vector_store=get_vector_store_cached(),
        metadata_store=get_metadata_store_cached(),
        embedding_model=get_embedding_model_cached(),
        config=config
    )

def validate_component(component, component_name: str) -> bool:
    """Validate that a component has required methods"""
    try:
        if component is None:
            return False
        
        # Add specific validation based on component type
        if component_name == "metadata_store":
            required_methods = ['get_stats', 'get_all_documents']
        elif component_name == "rag_pipeline":
            required_methods = ['query']
        else:
            return True  # Skip validation for other components
            
        for method in required_methods:
            if not hasattr(component, method):
                logger.error(f"{component_name} missing required method: {method}")
                return False
        return True
    except Exception as e:
        logger.error(f"Error validating {component_name}: {e}")
        return False

def initialize_system() -> bool:
    """Initialize the RAG system components with comprehensive error handling"""
    try:
        with st.spinner("Initializing system components..."):
            # Initialize cached components
            logger.info("Initializing embedding model (cached)...")
            embedding_model = get_embedding_model_cached()
            logger.info("Initializing vector store (cached)...")
            vector_store = get_vector_store_cached()
            logger.info("Initializing metadata store (cached)...")
            metadata_store = get_metadata_store_cached()
            if not validate_component(metadata_store, "metadata_store"):
                raise Exception("Metadata store validation failed")
            
            # Initialize RAG pipeline
            logger.info("Initializing RAG pipeline (cached)...")
            rag_pipeline = get_rag_pipeline_cached()
            if not validate_component(rag_pipeline, "rag_pipeline"):
                raise Exception("RAG pipeline validation failed")
            
            # Store in session state
            st.session_state.vector_store = vector_store
            st.session_state.metadata_store = metadata_store
            st.session_state.rag_pipeline = rag_pipeline
            st.session_state.system_initialized = True
            st.session_state.initialization_error = None
            
            st.success("System initialized successfully!")
            logger.info("System initialization completed successfully")
            return True
            
    except Exception as e:
        error_msg = f"Failed to initialize system: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        st.error(error_msg)
        st.session_state.initialization_error = str(e)
        st.session_state.system_initialized = False
        return False

def safe_get_stats():
    """Safely get metadata store statistics"""
    try:
        if st.session_state.metadata_store:
            return st.session_state.metadata_store.get_stats()
        return {'total_documents': 0, 'total_chunks': 0}
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {'total_documents': 0, 'total_chunks': 0}

def safe_get_documents():
    """Safely get document list"""
    try:
        if st.session_state.metadata_store:
            return st.session_state.metadata_store.get_all_documents()
        return []
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        return []

def safe_query(prompt: str):
    """Safely query the RAG pipeline"""
    try:
        if st.session_state.rag_pipeline:
            return st.session_state.rag_pipeline.query(prompt)
        raise Exception("RAG pipeline not available")
    except Exception as e:
        logger.error(f"Error in RAG query: {e}")
        raise

def main():
    # Title and description
    st.title("🚨 Disaster Preparedness Chatbot")
    st.markdown("""
    Welcome to the Disaster Preparedness Chatbot! This AI-powered tool provides accurate, 
    context-specific information about disaster preparedness and resilience from expert-curated documents.
    
    **How it works:**
    Ask questions about disaster preparedness and get accurate answers!
    """)
    
    # Show initialization error if any
    if st.session_state.initialization_error:
        st.error(f"System initialization failed: {st.session_state.initialization_error}")
        if st.button("Retry Initialization"):
            st.session_state.initialization_error = None
            initialize_system()
    
    # Auto-initialize system on first load
    if not st.session_state.system_initialized and not st.session_state.initialization_error:
        initialize_system()
    
    # Sidebar for knowledge base info
    with st.sidebar:
        st.header("📚 Knowledge Base Information")
        
        if st.session_state.system_initialized:
            try:
                # Display current knowledge base stats
                stats = safe_get_stats()
                st.metric("Documents", stats.get('total_documents', 0))
                st.metric("Total Chunks", stats.get('total_chunks', 0))
                
                # Document list
                st.subheader("📄 Available Documents")
                documents = safe_get_documents()
                if documents:
                    for doc in documents:
                        try:
                            with st.expander(f"📄 {doc.get('filename', 'Unknown')}"):
                                st.write(f"**Added:** {doc.get('upload_date', 'Unknown')}")
                                st.write(f"**Size:** {doc.get('file_size', 0):,} bytes")
                                st.write(f"**Chunks:** {doc.get('num_chunks', 0)}")
                        except Exception as e:
                            logger.error(f"Error displaying document {doc}: {e}")
                            st.error(f"Error displaying document")
                else:
                    st.info("No documents in knowledge base yet.")
                    st.markdown("**Administrator:** Run `python database_setup.py` to load documents.")
            except Exception as e:
                st.error(f"Error loading sidebar information: {e}")
                logger.error(f"Sidebar error: {e}")
    
    # Main chat interface
    if st.session_state.system_initialized:
        st.header("❓ Ask About Disaster Preparedness")
        
        # Simple chat input without history
        if prompt := st.chat_input("Ask me about disaster preparedness..."):
            # Display user question
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display response
            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base..."):
                    try:
                        result = safe_query(prompt)
                        if isinstance(result, dict):
                            st.markdown(result.get('response', ''))
                            sources = result.get('sources', [])
                            if sources:
                                st.caption("Sources: " + ", ".join(sources))
                        else:
                            st.markdown(str(result))
                        
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error while processing your question: {str(e)}"
                        st.error(error_msg)
                        logger.error(f"Chat error: {e}")
    
    else:
        st.info("The system is not initialized. Check configuration or try reloading the page.")
        st.markdown("**Administrator:** Ensure S3 ingestion has been run via `python database_setup.py`")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Important:** This chatbot provides information based on expert-curated disaster preparedness documents. 
    Always verify critical information with official sources and emergency services.
    """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application crashed: {e}")
        logger.error(f"Application crash: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
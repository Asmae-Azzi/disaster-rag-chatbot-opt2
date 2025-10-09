import streamlit as st
import traceback
import logging
from database_setup import main as setup_knowledge_base

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration (must be the first Streamlit call)
st.set_page_config(
    page_title="Disaster Preparedness Chatbot",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global styles (custom theme + chat bubbles)
st.markdown(
    """
    <style>
    :root {
        --azy-primary: #0ea5e9; /* cyan-500 */
        --azy-primary-dark: #0284c7; /* cyan-600 */
        --azy-bg: #0b1220; /* dark slate */
        --azy-card: #111827; /* slate-900 */
        --azy-text: #e5e7eb; /* gray-200 */
        --azy-muted: #9ca3af; /* gray-400 */
    }
    .azy-hero {
        background: radial-gradient(1200px 400px at 10% -10%, rgba(14,165,233,0.25), transparent),
                    radial-gradient(1200px 400px at 90% -10%, rgba(99,102,241,0.2), transparent);
        padding: 24px 24px 8px 24px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 8px;
    }
    .azy-title {
        display:flex; align-items:center; gap:12px; font-weight:800; font-size: 26px; color: var(--azy-text);
    }
    .azy-badge { 
        display:inline-block; padding: 4px 10px; border-radius: 999px; 
        background: linear-gradient(90deg, var(--azy-primary), #6366f1);
        color: white; font-size: 12px; font-weight: 700; letter-spacing: 0.3px;
    }
    .azy-sub { color: var(--azy-muted); margin-top: 6px; font-size: 14px; }
    .azy-card {
        background: var(--azy-card); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 12px;
    }
    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        background: var(--azy-card) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 12px !important;
        padding: 12px 14px !important;
        margin-bottom: 10px !important;
    }
    [data-testid="stChatMessage"] p { color: var(--azy-text) !important; }
    /* Example prompt chips */
    .azy-chip { 
        display:inline-block; margin: 4px 6px 0 0; padding: 6px 10px; border-radius: 999px; 
        background: rgba(14,165,233,0.12); color: var(--azy-text); border: 1px solid rgba(14,165,233,0.35);
        font-size: 12px; cursor: pointer;
    }
    .azy-chip:hover { background: rgba(14,165,233,0.2); }
    /* Footer */
    .azy-footer { color: var(--azy-muted); font-size: 12px; text-align:center; margin-top: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def is_knowledge_base_empty():
    try:
        # Use cached factories to avoid duplicate heavy initializations
        vector_store = get_vector_store_cached()
        info = vector_store.get_collection_info()
        return info.get('document_count', 0) == 0
    except Exception as e:
        logger.error(f"Error checking knowledge base emptiness: {e}")
        return True
        

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
        elif component_name == "vector_store":
            required_methods = ['get_collection_info']
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
            if not validate_component(vector_store, "vector_store"):
                raise Exception("Vector store validation failed")
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
            
            # Hide success toast to keep UI clean on deploy
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
    # --- DEBUG: Log collection info at startup (uses cached resources) ---
    try:
        info = get_vector_store_cached().get_collection_info()
        logger.info(f"Collection info at app startup: {info}")
    except Exception as e:
        logger.warning(f"Unable to fetch collection info at startup: {e}")
    
    # Title and description (hero header)
    st.markdown(
        """
        <div class="azy-hero">
            <div class="azy-title">🚨 <span>Azy — Disaster Preparedness Chatbot</span> <span class="azy-badge">LIVE</span></div>
            <div class="azy-sub">Ask questions and get concise, source-grounded answers from expert-curated documents.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    if is_knowledge_base_empty():
        # Auto-initialize knowledge base silently on first deploy
        with st.spinner("Preparing knowledge base..."):
            success = setup_knowledge_base()
        if not success:
            st.error("Failed to load knowledge base. Please check logs.")
            return  # Stop further execution if setup fails
        
    # Show initialization error if any
    if st.session_state.initialization_error:
        st.error(f"System initialization failed: {st.session_state.initialization_error}")
        if st.button("Retry Initialization"):
            st.session_state.initialization_error = None
            initialize_system()
    
    # Auto-initialize system on first load
    if not st.session_state.system_initialized and not st.session_state.initialization_error:
        initialize_system()
    
    # Sidebar removed: hide KB stats and documents from UI
    
    # Main chat interface
    if st.session_state.system_initialized:
        st.header("❓ Ask Azy")

        # Example prompts as quick chips
        examples = [
            "What should be in a 72-hour emergency kit?",
            "How do I prepare for a flood?",
            "Steps to take during an earthquake",
            "How to make a family emergency plan?",
        ]
        chip_html = " ".join([f'<span class="azy-chip" onclick="window.parent.postMessage({{type:\'azy_prompt\', value:\'{e}\'}}, \"*\")">{e}</span>' for e in examples])
        st.markdown(f"<div class='azy-card'>{chip_html}</div>", unsafe_allow_html=True)

        # Handle example chip clicks via components events (Streamlit reruns won’t catch JS; support via buttons too)
        cols = st.columns(len(examples))
        queued_prompt = None
        for i, e in enumerate(examples):
            if cols[i].button(e, key=f"azy_btn_{i}"):
                queued_prompt = e
        
        # Simple chat input without history + queued examples
        user_input = st.chat_input("Ask me about disaster preparedness...")
        prompt = queued_prompt or user_input
        if prompt:
            # Display user question
            with st.chat_message("user", avatar="🧑"):
                st.markdown(prompt)
            
            # Generate and display response
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Azy is searching the knowledge base..."):
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
        # Avoid Streamlit UI calls in CLI context
        logger.error(f"Application crash: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
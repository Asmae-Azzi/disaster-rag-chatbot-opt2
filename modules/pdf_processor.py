import fitz  # PyMuPDF
import PyPDF2 # Fallback for PyMuPDF
from pathlib import Path
from typing import List, Dict, Any, Union, IO, Optional
import re
import logging
import io # For BytesIO

# Import LangChain's text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import Config from parent directory
try:
    from config import Config
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config import Config

class PDFProcessor:
    """PDF processing, text extraction, and chunking"""
    
    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        """Initialize PDF processor with chunking parameters"""
        config = Config()
        self.chunk_size = chunk_size if chunk_size is not None else config.get_processing_config()['chunk_size']
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else config.get_processing_config()['chunk_overlap']
        
        # Initialize LangChain's text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def process_pdf(self, pdf_input: Union[str, bytes, IO[bytes]], filename: str = None) -> List[Dict[str, Any]]:
        """
        Process a PDF file (from path, bytes, or file-like object) and return chunked content.
        
        Args:
            pdf_input: Path to PDF (str), raw PDF bytes, or a file-like object (example: BytesIO)
            filename: Original filename (important for source tracking)
            
        Returns:
            List of chunked content with metadata.
        """
        try:
            if filename is None:
                if isinstance(pdf_input, str):
                    filename = Path(pdf_input).name
                else:
                    filename = "unknown_pdf_source.pdf" # Default for byte/stream input

            text_content = ""
            if isinstance(pdf_input, str): # Input is a file path
                text_content = self._extract_text_pymupdf(pdf_input)
                if not text_content:
                    text_content = self._extract_text_pypdf2(pdf_input)
            elif isinstance(pdf_input, bytes): # Input is raw bytes
                text_stream = io.BytesIO(pdf_input)
                text_content = self._extract_text_pymupdf_stream(text_stream)
                if not text_content:
                    text_stream.seek(0) # Reset stream for PyPDF2
                    text_content = self._extract_text_pypdf2_stream(text_stream)
            elif hasattr(pdf_input, 'read') and hasattr(pdf_input, 'seek'): # Input is a file-like object (e.g., BytesIO from S3)
                text_content = self._extract_text_pymupdf_stream(pdf_input)
                if not text_content:
                    pdf_input.seek(0) # Reset stream for PyPDF2
                    text_content = self._extract_text_pypdf2_stream(pdf_input)
            else:
                raise ValueError("Unsupported pdf_input type. Must be path, bytes, or file-like object.")

            if not text_content:
                raise Exception("No text content could be extracted from the PDF")
            
            # Clean and preprocess text
            cleaned_text = self._clean_text(text_content)
            
            # Split into chunks using LangChain's splitter
            chunks = self._create_chunks_langchain(cleaned_text, filename)
            
            self.logger.info(f"Successfully processed {filename}: {len(chunks)} chunks created.")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error processing PDF {filename}: {str(e)}")
            return [] # Return empty list on error

    def _extract_text_pymupdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF (fitz) from a file path."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            self.logger.debug(f"Extracted text using PyMuPDF from path: {pdf_path}")
            return text
        except Exception as e:
            self.logger.warning(f"PyMuPDF failed to extract text from path {pdf_path}: {str(e)}")
            return ""

    def _extract_text_pymupdf_stream(self, pdf_stream: IO[bytes]) -> str:
        """Extract text from PDF using PyMuPDF (fitz) from a file-like object."""
        try:
            doc = fitz.open(stream=pdf_stream.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            self.logger.debug("Extracted text using PyMuPDF from stream.")
            return text
        except Exception as e:
            self.logger.warning(f"PyMuPDF failed to extract text from stream: {str(e)}")
            return ""

    def _extract_text_pypdf2(self, pdf_path: str) -> str:
        """Extract text from PDF using PyPDF2 from a file path (fallback)."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or "" # Handle None from extract_text
            self.logger.debug(f"Extracted text using PyPDF2 from path: {pdf_path}")
            return text
        except Exception as e:
            self.logger.warning(f"PyPDF2 failed to extract text from path {pdf_path}: {str(e)}")
            return ""

    def _extract_text_pypdf2_stream(self, pdf_stream: IO[bytes]) -> str:
        """Extract text from PDF using PyPDF2 from a file-like object (fallback)."""
        try:
            # PyPDF2.PdfReader can take a file-like object directly
            reader = PyPDF2.PdfReader(pdf_stream)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            self.logger.debug("Extracted text using PyPDF2 from stream.")
            return text
        except Exception as e:
            self.logger.warning(f"PyPDF2 failed to extract text from stream: {str(e)}")
            return ""

    def _clean_text(self, text: str) -> str:
        """Clean and preprocess extracted text."""
        # Remove excessive whitespace and normalize newlines
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _create_chunks_langchain(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """Create chunks using LangChain's RecursiveCharacterTextSplitter."""
        chunks = []
        # LangChain's splitter returns Document objects
        documents = self.text_splitter.create_documents([text])
        
        for i, doc in enumerate(documents):
            chunks.append({
                'content': doc.page_content,
                'chunk_index': i,
                'source': filename,
                'chunk_size': len(doc.page_content),
                'page_number': doc.metadata.get('page_number', 0) # Placeholder, actual page number extraction is complex
            })
        self.logger.debug(f"Created {len(chunks)} chunks using LangChain splitter for {filename}.")
        return chunks


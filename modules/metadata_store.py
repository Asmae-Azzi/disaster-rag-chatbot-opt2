import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

class MetadataStore:
    """SQLite-based metadata store for document information"""
    
    def __init__(self, db_path: str = "./data/metadata.db"):
        """Initialize SQLite metadata store"""
        self.db_path = Path(db_path)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Create data directory if it doesn't exist
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize the database schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create documents table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT UNIQUE NOT NULL,
                        file_size INTEGER NOT NULL,
                        num_chunks INTEGER NOT NULL,
                        upload_date TEXT NOT NULL,
                        last_updated TEXT NOT NULL
                    )
                """)
                
                # Create chunks table for detailed chunk information
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chunks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id INTEGER NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        chunk_size INTEGER NOT NULL,
                        page_number INTEGER,
                        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                    )
                """)
                conn.commit()
                self.logger.info(f"Metadata database initialized at {self.db_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize metadata database: {str(e)}")
            raise Exception(f"Failed to initialize metadata database: {str(e)}")

    def add_document(self, filename: str, file_size: int, num_chunks: int) -> int:
        """
        Add a new document to the store or update an existing one.
        
        Args:
            filename: Name of the document file.
            file_size: Size of the document in bytes.
            num_chunks: Number of chunks the document was split into.
            
        Returns:
            The ID of the added/updated document.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                now = datetime.now().isoformat()
                
                # Check if document already exists
                cursor.execute("SELECT id FROM documents WHERE filename = ?", (filename,))
                result = cursor.fetchone()
                
                if result:
                    document_id = result[0]
                    # Update existing document
                    cursor.execute("""
                        UPDATE documents
                        SET file_size = ?, num_chunks = ?, last_updated = ?
                        WHERE id = ?
                    """, (file_size, num_chunks, now, document_id))
                    self.logger.info(f"Updated document metadata for '{filename}' (ID: {document_id})")
                else:
                    # Add new document
                    cursor.execute("""
                        INSERT INTO documents (filename, file_size, num_chunks, upload_date, last_updated)
                        VALUES (?, ?, ?, ?, ?)
                    """, (filename, file_size, num_chunks, now, now))
                    document_id = cursor.lastrowid
                    self.logger.info(f"Added new document metadata for '{filename}' (ID: {document_id})")
                
                conn.commit()
                return document_id
                
        except Exception as e:
            self.logger.error(f"Failed to add/update document metadata: {str(e)}")
            raise Exception(f"Failed to add/update document metadata: {str(e)}")

    def get_document(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by filename"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row # Return rows as dictionaries
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM documents WHERE filename = ?", (filename,))
                result = cursor.fetchone()
                
                return dict(result) if result else None
                
        except Exception as e:
            self.logger.error(f"Failed to get document metadata: {str(e)}")
            raise Exception(f"Failed to get document metadata: {str(e)}")

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all document metadata"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM documents ORDER BY upload_date DESC")
                results = cursor.fetchall()
                
                return [dict(row) for row in results]
                
        except Exception as e:
            self.logger.error(f"Failed to get all document metadata: {str(e)}")
            raise Exception(f"Failed to get all document metadata: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics about the stored documents and chunks"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM documents")
                total_documents = cursor.fetchone()[0]
                
                cursor.execute("SELECT SUM(num_chunks) FROM documents")
                total_chunks = cursor.fetchone()[0] or 0 # Handle case where sum is None
                
                cursor.execute("SELECT SUM(file_size) FROM documents")
                total_size_bytes = cursor.fetchone()[0] or 0
                
                return {
                    'total_documents': total_documents,
                    'total_chunks': total_chunks,
                    'total_size_bytes': total_size_bytes
                }
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {str(e)}")
            raise Exception(f"Failed to get database stats: {str(e)}")

    def delete_document(self, filename: str) -> bool:
        """Delete a document and its associated chunks from the store"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get document ID first
                cursor.execute("SELECT id FROM documents WHERE filename = ?", (filename,))
                result = cursor.fetchone()
                
                if result:
                    document_id = result[0]
                    
                    # Delete chunks first (foreign key constraint)
                    cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                    
                    # Delete document
                    cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
                    
                    conn.commit()
                    self.logger.info(f"Deleted document '{filename}' (ID: {document_id}) and its chunks.")
                    return True
                
                self.logger.warning(f"Attempted to delete non-existent document: '{filename}'")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to delete document: {str(e)}")
            raise Exception(f"Failed to delete document: {str(e)}")
    
    def get_document_chunks(self, filename: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT c.* FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE d.filename = ?
                    ORDER BY c.chunk_index
                """, (filename,))
                
                results = cursor.fetchall()
                return [dict(row) for row in results]
                
        except Exception as e:
            self.logger.error(f"Failed to get document chunks: {str(e)}")
            raise Exception(f"Failed to get document chunks: {str(e)}")

import os
import sqlite3
import hashlib
import pickle
import numpy as np
import faiss
from datetime import datetime
from pypdf import PdfReader

DB_PATH = 'data/rag.db'

class VectorManager:
    def __init__(self):
        os.makedirs('data/collections', exist_ok=True)
        # Initialize DB
        with self._get_db() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS collections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    collection_id INTEGER,
                    filename TEXT,
                    file_hash TEXT,
                    faiss_start_idx INTEGER,
                    faiss_end_idx INTEGER,
                    status TEXT,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (collection_id) REFERENCES collections(id)
                )
            ''')
            conn.commit()

    def _get_db(self):
        # We create a new connection per request to avoid multi-threading issues in Streamlit
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        return sqlite3.connect(DB_PATH)

    def get_collections(self):
        with self._get_db() as conn:
            cursor = conn.execute('SELECT id, name FROM collections ORDER BY name')
            return [{"id": row[0], "name": row[1]} for row in cursor.fetchall()]

    def create_collection(self, name):
        with self._get_db() as conn:
            try:
                conn.execute('INSERT INTO collections (name) VALUES (?)', (name,))
                conn.commit()
                # Create folders for the new collection
                row = conn.execute('SELECT id FROM collections WHERE name = ?', (name,)).fetchone()
                if row:
                    collection_id = row[0]
                    os.makedirs(f'data/collections/{collection_id}/documents', exist_ok=True)
                    os.makedirs(f'data/collections/{collection_id}/index', exist_ok=True)
                return True, "Collection created successfully."
            except sqlite3.IntegrityError:
                return False, "Collection already exists."

    def get_collection_id(self, name):
        with self._get_db() as conn:
            row = conn.execute('SELECT id FROM collections WHERE name = ?', (name,)).fetchone()
            return row[0] if row else None

    def _compute_hash(self, file_bytes):
        return hashlib.sha256(file_bytes).hexdigest()

    def process_file(self, collection_name, file_name, file_bytes, embedding_model, progress_callback=None):
        collection_id = self.get_collection_id(collection_name)
        if not collection_id:
            return False, "Collection not found."

        file_hash = self._compute_hash(file_bytes)

        # Check for duplication within the same collection
        with self._get_db() as conn:
            exists = conn.execute('SELECT id FROM documents WHERE collection_id = ? AND file_hash = ?', 
                                (collection_id, file_hash)).fetchone()
            if exists:
                return False, "File already exists in this collection."

            # Register as processing
            cursor = conn.cursor()
            cursor.execute('INSERT INTO documents (collection_id, filename, file_hash, status) VALUES (?, ?, ?, ?)', 
                           (collection_id, file_name, file_hash, 'processing'))
            doc_id = cursor.lastrowid
            conn.commit()

        try:
            # 1. Save PDF locally
            pdf_path = f'data/collections/{collection_id}/documents/{file_name}'
            with open(pdf_path, 'wb') as f:
                f.write(file_bytes)

            if progress_callback:
                progress_callback(0.1, "Extracting text from PDF...")

            # 2. Extract Text
            text = self._extract_text(pdf_path)

            if progress_callback:
                progress_callback(0.3, "Splitting into chunks...")

            # 3. Text Chunking
            chunks = self._split_text(text, chunk_size=600, overlap=100)

            if progress_callback:
                progress_callback(0.4, "Generating embeddings...")

            # 4. Create Embeddings in Batches
            embeddings = []
            batch_size = 32
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_embeddings = embedding_model.encode(batch, show_progress_bar=False)
                embeddings.extend(batch_embeddings)
                
                if progress_callback:
                    prog = 0.4 + 0.4 * (min(i + batch_size, len(chunks)) / len(chunks))
                    progress_callback(prog, f"Embedding chunks... ({min(i + batch_size, len(chunks))}/{len(chunks)})")

            embeddings_array = np.array(embeddings).astype('float32')

            if progress_callback:
                progress_callback(0.85, "Updating FAISS Vector Store...")

            # 5. Append to FAISS Index
            index_dir = f'data/collections/{collection_id}/index'
            faiss_path = f'{index_dir}/faiss_index.bin'
            chunks_path = f'{index_dir}/chunks.pkl'

            if os.path.exists(faiss_path) and os.path.exists(chunks_path):
                index = faiss.read_index(faiss_path)
                with open(chunks_path, 'rb') as f:
                    existing_chunks = pickle.load(f)
            else:
                # Initialize new FAISS index
                dimension = embeddings_array.shape[1]
                index = faiss.IndexFlatL2(dimension)
                existing_chunks = []

            start_idx = index.ntotal
            index.add(embeddings_array)
            existing_chunks.extend(chunks)
            end_idx = index.ntotal - 1

            # 6. Save Store
            faiss.write_index(index, faiss_path)
            with open(chunks_path, 'wb') as f:
                pickle.dump(existing_chunks, f)

            if progress_callback:
                progress_callback(0.95, "Updating Database Metadata...")

            # 7. Update SQLite specific ranges and finalize
            with self._get_db() as conn:
                conn.execute('''
                    UPDATE documents 
                    SET faiss_start_idx = ?, faiss_end_idx = ?, status = ?
                    WHERE id = ?
                ''', (start_idx, end_idx, 'completed', doc_id))
                conn.commit()

            if progress_callback:
                progress_callback(1.0, "Processing successful!")

            return True, "File processed successfully."

        except Exception as e:
            # Mark as failed in SQLite
            with self._get_db() as conn:
                conn.execute('UPDATE documents SET status = ? WHERE id = ?', (f'failed: {str(e)}', doc_id))
                conn.commit()
            return False, f"Processing failed: {str(e)}"

    def _extract_text(self, pdf_path):
        reader = PdfReader(pdf_path)
        text = ""
        for page_num, page in enumerate(reader.pages):
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.extract_text()
        return text

    def _split_text(self, text, chunk_size, overlap):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start += chunk_size - overlap
        return chunks

    def load_index_for_collection(self, collection_name):
        collection_id = self.get_collection_id(collection_name)
        if not collection_id:
            return None, None
            
        faiss_path = f'data/collections/{collection_id}/index/faiss_index.bin'
        chunks_path = f'data/collections/{collection_id}/index/chunks.pkl'

        if not os.path.exists(faiss_path) or not os.path.exists(chunks_path):
            return None, None

        index = faiss.read_index(faiss_path)
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)
            
        return index, chunks

    def retrieve(self, query, index, chunks, embedding_model, k=3):
        if not index or not chunks:
            return [], []
            
        query_embedding = embedding_model.encode([query])
        distances, indices = index.search(np.array(query_embedding).astype('float32'), k)
        
        relevant_chunks = []
        ret_distances = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks):
                relevant_chunks.append(chunks[idx])
                ret_distances.append(distances[0][i])
                
        return relevant_chunks, ret_distances

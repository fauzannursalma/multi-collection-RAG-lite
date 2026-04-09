# Medical RAG Chatbot — Recap Update

Catatan perubahan harian pada project Medical RAG Chatbot.

====================================================================
## 8 April 2026
====================================================================

### Phase 1: Multi-Collection RAG System Architecture
- **Dynamic Storage Architecture:** Merombak sistem dari single static vector store menjadi Multi-Collection RAG. Aplikasi menggunakan struktur folder dinamis di `data/collections/<collection_id>/` untuk memisahkan indeks FAISS per konteks/kategori pengetahuan.
- **SQLite Metadata Management:** Menambahkan database internal `rag.db` untuk mencatat metadata koleksi dokumen, melacak rentang indeks chunk (`faiss_start_idx` dan `faiss_end_idx`), dan otomatis memblokir redudansi upload file berdasarkan validasi hash SHA-256 (`file_hash`).
- **Core Logic Refactoring (`VectorManager`):** Menghapus skrip lawas `process_documents.py` dan memusatkan fungsionalitas embedding (parsing PDF pypdf, SentenceTransformer chunking, dan FAISS indexing) di dalam satu kelas modular `VectorManager` yang interaktif (di `vector_manager.py`) serta sangat efisien melalui metode penambahan incremental `faiss.add()`.

### Phase 2: UI/UX & Streamlit Application Enhancements
- **Drag-and-Drop Bulk Upload:** Memperbarui antarmuka `st.file_uploader` pada sidebar menjadi `accept_multiple_files=True`. Pengguna kini bisa melempar (drag-and-drop) banyak file PDF sekaligius dan menembaknya ke model dalam satu putaran klik.
- **Combined Progress Tracking:** Memodifikasi rendering hook callback `st.progress` agar mampu mendistribusikan animasi kemajuan secara akurat dengan menghitung sisa fraksi tugas di saat array multipel dokumen diproses oleh Transformer.
- **Isolated Contextual Memory:** Menerapkan arsitektur baru bagi Streamlit Session State yakni memori dipilah berbasis key koleksi (`st.session_state.messages[active_collection]`). Ini menjaga percakapan di koleksi A tidak bercampur halusinasi dengan dokumen di koleksi B.
- **Dynamic Prompt Generative AI:** Template prompt Gemini 2.5 dihapus sifat _hardcode_-nya dan sekarang diinjeksi via nama koleksi saat ini (`active_collection`), memastikan AI menyesuaikan nadanya berdasarkan ruang data yang terpilih.
- **Strict Execution Prompt (Update):** Merombak total `system prompt` di `app.py` agar jauh lebih disiplin. AI kini diwajibkan mematuhi pedoman *Language Matching* (membalas sesuai bahasa *User Query*), *Strict Grounding* (menolak rujukan eksternal dan wajib menolak menjawab/fallback jika tidak ditemukan di teks), serta menjaga format lebih rapi terstruktur (bullet points & citations).
- **Environment Bugfix (`ModuleNotFoundError`):** Menambahkan package `torchvision` ke dalam `requirements.txt` dan menginstalnya guna membendung error dari engine image-loader tersembunyi `zoedepth` yang diminta sistem dependensi transformer saat Streamlit dihidupkan.

**Files Created & Deleted:**
- `[NEW]` `vector_manager.py` — Pengatur fungsionalitas embedding backend.
- `[DEL]` `process_documents.py` — Pemroses index lama di-deprecated.
- `[DEL]` Direkori `vector_store/` kuno terhapus sepenuhnya.

**Files Modified:**
- `app.py`, `requirements.txt`, `DOCUMENTATION.md`

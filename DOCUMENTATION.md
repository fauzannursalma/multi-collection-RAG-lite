# Multi-Collection RAG Lite Documentation

## 1. Project Overview
Multi-Collection RAG Lite is an intelligent assistant designed to answer dynamic questions based on uploaded knowledge bases. Upgraded from a static single-document RAG, it now uses a Multi-Collection architecture allowing users to construct segregated directories (e.g., Medical, Finance, Legal), bulk-upload PDF documents via drag-and-drop, and converse with Google Generative AI (Gemini) whose logic is grounded securely within the selected active collection.

## 2. Key Features
- **Dynamic Multi-Collection Capabilities**: Create specific topic collections directly from the UI and switch between them, ensuring chat history and context remain heavily isolated and relevant to the subject.
- **Advanced Bulk Document Pipeline**: Drag and drop multiple PDF documents simultaneously. The system seamlessly handles batch extraction, chunking, and appends new high-dimensional embeddings incrementally.
- **SQLite Metadata Tracker**: All file uploads, hashing records (preventing duplicate processing), and collection hierarchies are robustly tracked using an internal SQLite database (`rag.db`).
- **Local Vector Database**: Built upon **FAISS** (Facebook AI Similarity Search) for blazingly fast context chunk retrieval natively stored per collection.
- **Generative AI Integration**: Powered by **Google's Gemini 2.5 Flash** model with dynamic prompt-injections to generate confident responses based safely on the provided FAISS references.
- **Interactive Progress UI**: The Streamlit interface displays real-time embedding compilation animations, and offers source references so users know exactly *where* the AI pulled facts from.

## 3. Technology Stack
- **Frontend / UI**: [Streamlit](https://streamlit.io/)
- **Embedding Backend Manager**: `vector_manager.py` (Object-Oriented Abstraction)
- **Embedding Model**: `all-MiniLM-L6-v2` via [Sentence-Transformers](https://www.sbert.net/)
- **Vector Store Database**: [FAISS CPU](https://github.com/facebookresearch/faiss) alongside [SQLite](https://www.sqlite.org/).
- **Generative LLM**: [Google Generative AI (Gemini)](https://ai.google.dev/)
- **PDF Processing**: `pypdf`

## 4. Project Structure
```text
.
├── .env                        # Environment variables (e.g., GEMINI_API_KEY)
├── app.py                      # Main Streamlit application
├── vector_manager.py           # Core logic handling SQLite DB, Chunking, and FAISS Indexes
├── test_rag.py                 # Testing script for offline operations
├── requirements.txt            # Python package dependencies
├── recap_update.md             # Developer logs outlining version changes
└── data/                       # Dynamic storage directory
    ├── rag.db                  # Relational SQLite metadata tracking tables
    └── collections/            # Root folder for specific collections
        └── <identifier>/       # Dynamically generated folder for each collection
            ├── documents/      # The raw copied PDF files deposited by the user
            └── index/          # The faiss_index.bin and chunks.pkl specific to this collection
```

## 5. System Architecture
The application runs as a cohesive live application governed heavily by user actions on the frontend:

1. **Initialization**: On boot-up, `app.py` triggers the `VectorManager` instance which validates the directory map and establishes connection loops with `rag.db`.
2. **Collection Switcher**: Users create or swap collections on the sidebar. This triggers Streamlit's engine to swap the active `faiss_index` in memory and refresh the localized chat session dictionary.
3. **Pipelined Uploads**: When PDFs are dropped in via the Streamlit drag-and-drop multi-uploader:
    - Files are hashed via SHA-256 preventing processing loops.
    - Raw PDFs are dropped into `collections/<id>/documents`.
    - Text blocks are cleanly chunked and batched identically to the `all-MiniLM` model dimension size.
    - Appended vectors (`faiss.add()`) update the store incrementally.
4. **Contextual Answering**: Questions are reduced to vectors, cross-referenced within the active collection's localized FAISS index, and subsequently forwarded alongside the chunks to the Gemini API formatted under dynamic conversational guardrails.

## 6. Installation & Setup

### Prerequisites
- Python 3.8+
- A Google Gemini API Key

### Setup Steps
1. **Clone or Navigate to the project directory**
   ```bash
   cd d:\project\ragchatbot
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   Create a `.env` file in the root directory and add your Gemini API Key:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```

## 7. Usage

Unlike the previous iteration, there are no offline preprocessing scripts needed! Everything is managed natively through the elegant GUI pipeline. 

Simply launch the Streamlit Application:
```bash
streamlit run app.py
```
- Open the web browser interface.
- From the sidebar, click **➕ Create New Collection** (Name it anything, e.g., 'Programming Guidelines').
- Select it under **📂 Select Collection**.
- Use the **Drag & Drop** area to upload multiple PDFs concurrently.
- Wait for the parsing and indexing progress bar to reach 100%.
- Start chatting contextually!

import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from vector_manager import VectorManager

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = {}  # Dict: collection_name -> [messages]

if "v_manager" not in st.session_state:
    st.session_state.v_manager = VectorManager()

# Page config
st.set_page_config(
    page_title="Multi-Collection RAG Chatbot Lite",
    page_icon="📚",
    layout="wide"
)

# Load embedding model globally in Streamlit cache cache_resource
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()
v_manager = st.session_state.v_manager

# Fetch Collections
collections = v_manager.get_collections()
collection_names = [c["name"] for c in collections]

# Title
st.title("📚 Multi-Collection RAG Chatbot Lite")
st.markdown("Create collections, upload documents, and chat with AI that knows your specific context.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Collection Creation
    with st.expander("➕ Create New Collection", expanded=False):
        new_col_name = st.text_input("Collection Name")
        if st.button("Create"):
            if new_col_name:
                success, msg = v_manager.create_collection(new_col_name)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.warning("Please enter a name.")

    # Collection Selection
    st.markdown("### 📂 Select Collection")
    if not collection_names:
        st.info("No collections available. Please create one.")
        active_collection = None
    else:
        active_collection = st.selectbox("Active Collection", collection_names)

    # Document Uploader
    if active_collection:
        st.markdown(f"### 📄 Upload to **{active_collection}**")
        uploaded_files = st.file_uploader("Upload PDFs (Drag & drop multiple files)", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            if st.button(f"Process {len(uploaded_files)} Document(s)"):
                # Progress setup
                progress_bar = st.progress(0.0, text="Initializing...")
                total_files = len(uploaded_files)
                
                for idx, file in enumerate(uploaded_files):
                    file_bytes = file.read()
                    
                    def update_progress(prog, text):
                        overall_prog = (idx + prog) / total_files
                        progress_bar.progress(overall_prog, text=f"[{idx+1}/{total_files}] Processing {file.name}: {text}")
                        
                    success, msg = v_manager.process_file(
                        active_collection, 
                        file.name, 
                        file_bytes, 
                        embedding_model, 
                        progress_callback=update_progress
                    )
                    
                    if success:
                        st.success(f"✅ {file.name}: {msg}")
                    else:
                        st.error(f"❌ {file.name}: {msg}")
                
                progress_bar.progress(1.0, text="🎉 All documents processed!")

    st.markdown("---")
    
    # Model configuration
    temperature = st.slider("Response Temperature", 0.0, 1.0, 0.7, 0.1)
    top_k = st.slider("Number of Retrieved Chunks", 1, 10, 3, 1)

    st.markdown("---")
    if active_collection and st.button("🗑️ Clear Chat History"):
        if active_collection in st.session_state.messages:
            st.session_state.messages[active_collection] = []
        st.rerun()


def generate_response(query, context_chunks, collection_name, temperature=0.7):
    """Generate response using Gemini API"""
    context = "\n\n".join([f"[Source {i + 1}]: {chunk}" for i, chunk in enumerate(context_chunks)])

    prompt = f"""You are a helpful assistant for the '{collection_name}' domain. Answer the question based ONLY on the provided context.

Context from {collection_name}:
{context}

Question: {query}

Instructions:
- Answer based only on the provided context
- If the context doesn't contain enough information, say so clearly
- Be concise and accurate
- If referencing specific information, mention which source ([Source 1], [Source 2], etc.)

Answer:"""

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=1024,
            )
        )
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Main chat interface
if active_collection:
    # Initialize history for this collection if missing
    if active_collection not in st.session_state.messages:
        st.session_state.messages[active_collection] = []
        
    messages = st.session_state.messages[active_collection]

    # Display chat messages
    for msg_idx, message in enumerate(messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 View Retrieved Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i + 1}** (Similarity: {source['score']:.4f})")
                        st.text_area(
                            f"source_{i}",
                            source["text"],
                            height=100,
                            label_visibility="collapsed",
                            key=f"source_msg{msg_idx}_chunk{i}"
                        )

    # Chat input
    if prompt := st.chat_input(f"Ask a question about {active_collection}..."):
        st.session_state.messages[active_collection].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                # Load index
                index, chunks = v_manager.load_index_for_collection(active_collection)
                
                if index is None or chunks is None or len(chunks) == 0:
                    st.warning("No documents have been indexed for this collection yet. Please upload a PDF.")
                    response = "I couldn't find any documents in this collection."
                    sources = []
                else:
                    relevant_chunks, distances = v_manager.retrieve(
                        prompt, index, chunks, embedding_model, k=top_k
                    )
                    
                    response = generate_response(prompt, relevant_chunks, active_collection, temperature)
                    
                    sources = [
                        {"text": chunk, "score": 1 / (1 + dist) if dist != 0 else 1.0}
                        for chunk, dist in zip(relevant_chunks, distances)
                    ]

                st.markdown(response)

                if sources:
                    with st.expander("📚 View Retrieved Sources"):
                        for i, source in enumerate(sources):
                            st.markdown(f"**Source {i + 1}** (Similarity: {source['score']:.4f})")
                            st.text_area(
                                f"source_{i}",
                                source["text"],
                                height=100,
                                label_visibility="collapsed",
                                key=f"source_current_msg{len(messages)}_chunk{i}"
                            )

                st.session_state.messages[active_collection].append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources
                })
else:
    st.info("👈 Please create or select a collection from the sidebar to start chatting.")
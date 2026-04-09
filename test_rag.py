"""
Test script to verify RAG pipeline is working correctly
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import time

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def load_vector_store():
    """Load the vector store and embedding model"""
    print("🔄 Loading vector store...")

    # Load embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load FAISS index
    index = faiss.read_index("vector_store/faiss_index.bin")

    # Load chunks
    with open("vector_store/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    # Load metadata
    with open("vector_store/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    print(f"✅ Loaded {metadata['num_chunks']} chunks")
    print(f"✅ Index contains {index.ntotal} vectors")

    return embedding_model, index, chunks, metadata

def test_retrieval(query, embedding_model, index, chunks, k=3):
    """Test the retrieval component"""
    print(f"\n📝 Query: {query}")
    print("-" * 60)

    # Embed query
    query_embedding = embedding_model.encode([query])

    # Search
    start_time = time.time()
    distances, indices = index.search(np.array(query_embedding).astype('float32'), k)
    search_time = time.time() - start_time

    print(f"⏱️  Search time: {search_time:.4f} seconds")
    print(f"\n🔍 Top {k} retrieved chunks:\n")

    retrieved_chunks = []
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        chunk = chunks[idx]
        score = 1 / (1 + dist)  # Convert distance to similarity score

        print(f"[Chunk {i+1}] Similarity: {score:.4f}")
        print(f"{chunk[:200]}...")
        print()

        retrieved_chunks.append(chunk)

    return retrieved_chunks

def test_generation(query, context_chunks):
    """Test the generation component"""
    print("🤖 Generating response with Gemini...")
    print("-" * 60)

    # Prepare context
    context = "\n\n".join([f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)])

    # Create prompt
    prompt = f"""You are a helpful medical assistant. Answer the question based ONLY on the provided context from the Medical Book.

Context from Medical Book:
{context}

Question: {query}

Instructions:
- Answer based only on the provided context
- If the context doesn't contain enough information, say so clearly
- Be concise and accurate
- Use medical terminology appropriately
- If referencing specific information, mention which source ([Source 1], [Source 2], etc.)

Answer:"""

    # Generate
    start_time = time.time()
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.7,
            max_output_tokens=1024,
        )
    )
    generation_time = time.time() - start_time

    print(f"⏱️  Generation time: {generation_time:.4f} seconds")
    print(f"\n💬 Response:\n")
    print(response.text)

    return response.text

def main():
    """Run tests"""
    print("="*60)
    print("🧪 Testing RAG Pipeline")
    print("="*60)

    # Check if vector store exists
    if not os.path.exists("vector_store/faiss_index.bin"):
        print("❌ Vector store not found!")
        print("Please run: python process_documents.py")
        return

    # Check if API key is set
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found in .env file!")
        return

    try:
        # Load vector store
        embedding_model, index, chunks, metadata = load_vector_store()

        # Test queries
        test_queries = [
            "What is Hemofiltration?",
            "Mechanism of COVID-19 infection?"
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*60}")
            print(f"Test {i}/{len(test_queries)}")
            print('='*60)

            # Test retrieval
            retrieved_chunks = test_retrieval(query, embedding_model, index, chunks, k=3)

            # Test generation
            response = test_generation(query, retrieved_chunks)

            print("\n")

        print("="*60)
        print("✅ All tests completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
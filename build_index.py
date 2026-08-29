"""
AgroRithm / AgroSentinel - RAG Index Builder
Embeds the local knowledge corpus into a local ChromaDB vector store.
Runs fully offline once the embedding model is downloaded the first time.
"""

import os
os.environ["HF_HUB_OFFLINE'] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import chromadb
from sentence_transformers import SentenceTransformer

CORPUS_DIR = "corpus"
DB_DIR = "chroma_db"

def chunk_text(text, chunk_size=500, overlap=50):
    """Simple sentence-aware chunking so retrieval returns focused passages."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            # split long paragraphs into smaller pieces
            words = para.split()
            current = []
            current_len = 0
            for word in words:
                current.append(word)
                current_len += len(word) + 1
                if current_len >= chunk_size:
                    chunks.append(" ".join(current))
                    current = []
                    current_len = 0
            if current:
                chunks.append(" ".join(current))
    return chunks

def main():
    print("Loading embedding model (small, local, offline after first download)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Setting up local ChromaDB...")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name="agrorithm_knowledge")

    doc_id = 0
    for filename in os.listdir(CORPUS_DIR):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(CORPUS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        print(f"  {filename}: {len(chunks)} chunks")

        for chunk in chunks:
            embedding = model.encode(chunk).tolist()
            collection.add(
                ids=[f"doc_{doc_id}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": filename}]
            )
            doc_id += 1

    print(f"\nDone. Indexed {doc_id} chunks from {CORPUS_DIR}/ into {DB_DIR}/")
    print("You can now run query_test.py to test retrieval.")

if __name__ == "__main__":
    main()

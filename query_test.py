"""
AgroRithm / AgroSentinel - RAG Retrieval Test
Quick way to confirm the vector index retrieves relevant context
before wiring it into the full LLM pipeline.
"""

import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = "chroma_db"

def main():
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name="agrorithm_knowledge")

    print("\nRAG retrieval test. Type a query (or 'exit' to quit).\n")
    while True:
        query = input("> ")
        if query.lower() in ("exit", "quit"):
            break

        query_embedding = model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        print("\n--- Top matching context ---")
        for i, doc in enumerate(results["documents"][0]):
            source = results["metadatas"][0][i]["source"]
            print(f"\n[{i+1}] (from {source})")
            print(doc)
        print("\n-----------------------------\n")

if __name__ == "__main__":
    main()

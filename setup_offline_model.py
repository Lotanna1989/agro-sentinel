"""
AgroRithm / AgroSentinel - One-Time Setup Script
Downloads and caches the local embedding model (all-MiniLM-L6-v2) used by the
RAG pipeline. Run this ONCE, with internet access, before using build_index.py,
rag_pipeline.py, or automated_pipeline.py for the first time.

After this completes, all other scripts run fully offline (they set
HF_HUB_OFFLINE=1 and load the model from the local cache created here).
"""

from sentence_transformers import SentenceTransformer

print("Downloading and caching embedding model (all-MiniLM-L6-v2)...")
print("This requires internet access and only needs to run once.\n")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("\nDone. The embedding model is now cached locally.")
print("You can now run build_index.py, rag_pipeline.py, and automated_pipeline.py")
print("fully offline, with no internet connection required.")

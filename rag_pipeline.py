"""
AgroRithm / AgroSentinel - Full RAG + LLM Pipeline
Retrieves relevant context from the local knowledge corpus, then sends it
to the locally running Qwen2.5 model (via llama-server) to generate a
grounded, farm-protection-focused answer.

Requires llama-server.exe running first:
    .\\llama-server.exe -hf Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M --port 8080
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import chromadb
import json
import requests
from sentence_transformers import SentenceTransformer

DB_DIR = "chroma_db"
LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"

SYSTEM_PROMPT = """You are AgroSentinel, an offline AI farm advisor for African smallholder farmers, herders, and forest guards.
You give practical advice on farm intrusions, crop protection, livestock health (heat stress, water quality, ammonia,
disease signs), and market/sale decisions.

Speak directly and naturally, like an experienced local advisor talking to a farmer - not like a system reporting on
its own data sources. Never say phrases like "based on the context provided," "the context does not contain," or
"according to the information given." Just give the advice directly, as if you already know it.

Rules:
- Give clear, confident, practical guidance suited to someone in the field.
- If you don't have a specific figure (like an exact current price), say so plainly and naturally
  ("I don't have today's exact price, but a typical range is...") rather than referring to "the context."
- Always state clearly WHO should be notified when relevant: the farm owner, a forest guard unit, or veterinary/
  agricultural extension services, based on the type and severity of the situation.
- Keep answers concise and actionable.
- Do not invent specific statistics, prices, or diagnoses you are not reasonably confident about - but express
  that uncertainty naturally, not by referencing "the context" or "the data."
"""

def retrieve_context(query, embed_model, collection, n_results=6):
    query_embedding = embed_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return chunks, sources

def build_prompt(query, chunks, sources):
    knowledge_block = ""
    for chunk in chunks:
        knowledge_block += f"\n{chunk}\n"
    return f"Relevant background knowledge you already know:\n{knowledge_block}\n\nFarmer's question: {query}\n\nAnswer the farmer directly and naturally."

def query_llm(user_prompt):
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 400,
        "stream": True
    }
    full_response = ""
    with requests.post(LLAMA_SERVER_URL, json=payload, timeout=120, stream=True) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if not decoded.startswith("data: "):
                continue
            data_str = decoded[len("data: "):]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    print(delta, end="", flush=True)
                    full_response += delta
            except (KeyError, ValueError):
                continue
    print()  # newline after streaming finishes
    return full_response

def main():
    print("Loading embedding model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name="agrorithm_knowledge")

    print("\nAgroSentinel RAG pipeline ready. Type a query (or 'exit' to quit).\n")
    print("Make sure llama-server.exe is running on port 8080 first!\n")

    while True:
        query = input("> ")
        if query.lower() in ("exit", "quit"):
            break

        print("\nRetrieving relevant context...")
        chunks, sources = retrieve_context(query, embed_model, collection)
        print(f"Retrieved {len(chunks)} chunks from: {', '.join(set(sources))}")

        prompt = build_prompt(query, chunks, sources)

        print("\nGenerating grounded response...\n")
        try:
            print("--- AgroSentinel Response ---")
            answer = query_llm(prompt)
            print("-----------------------------\n")
        except requests.exceptions.ConnectionError:
            print("\n[ERROR] Could not connect to llama-server on port 8080.")
            print("Make sure llama-server.exe is running in a separate terminal.\n")

if __name__ == "__main__":
    main()

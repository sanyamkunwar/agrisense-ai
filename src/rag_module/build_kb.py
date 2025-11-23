import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

load_dotenv()

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "models/embeddings/")
KB_PATH = os.getenv("KB_PATH", "data/knowledge_base/processed_kb.json")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

print(f"Loading embedding model: {EMBED_MODEL}")
embedder = SentenceTransformer(EMBED_MODEL)

client = PersistentClient(path=CHROMA_DB_DIR)

try:
    client.delete_collection("agri_kb")
except:
    pass

collection = client.create_collection("agri_kb")


def build_kb():
    print(f"Loading KB: {KB_PATH}")

    with open(KB_PATH, "r") as f:
        data = json.load(f)

    texts = [x["content"] for x in data]
    sources = [x["source"] for x in data]

    print(f"Total chunks: {len(texts)}")

    embeddings = embedder.encode(texts).tolist()

    print("Saving to Chroma…")
    for i, (emb, txt, src) in enumerate(zip(embeddings, texts, sources)):
        collection.add(
            ids=[str(i)],
            embeddings=[emb],
            documents=[txt],
            metadatas=[{"source": src}]
        )

    print("✅ KB Build Complete.")
    print("📦 Stored at:", CHROMA_DB_DIR)


if __name__ == "__main__":
    build_kb()

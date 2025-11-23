import os
from typing import List, Tuple
from dotenv import load_dotenv
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "models/embeddings/")

client = PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_collection("agri_kb")

embedder = SentenceTransformer(EMBED_MODEL)
llm = OpenAI(api_key=OPENAI_API_KEY)


def build_prompt(query: str, docs: List[str]) -> str:
    ctx = "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(docs)])

    return f"""
You are an agricultural expert.

User question:
{query}

Use ONLY this context:
{ctx}

Guidelines:
- Explain simply.
- Give treatment + prevention.
- Provide both organic and chemical options.
- Cite sources using [1], [2], [3].

If unsure, say so.
"""


def generate_answer(query: str, top_k: int = 5) -> Tuple[str, List[str]]:
    q_emb = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings=q_emb,
        n_results=top_k
    )

    docs = results["documents"][0]
    sources = results["metadatas"][0]

    prompt = build_prompt(query, docs)

    response = llm.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are an agriculture assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    return answer, [s["source"] for s in sources]

import os
import json
from .utils import clean_text, chunk_text, extract_text_from_pdf

KB_DIR = "data/knowledge_base/"
OUTPUT_JSON = "data/knowledge_base/processed_kb.json"


def safe_read_pdf(path):
    try:
        return extract_text_from_pdf(path)
    except Exception as e:
        print(f"[WARNING] PDF failed for {os.path.basename(path)}: {e}")
        return None


def safe_read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except:
        return None


def ingest_documents():
    kb_entries = []

    files = sorted(os.listdir(KB_DIR))
    print(f"📄 Found {len(files)} files.\n")

    for fname in files:
        path = os.path.join(KB_DIR, fname)
        if not os.path.isfile(path):
            continue

        print(f"➡ Processing {fname} ...")

        text = None
        if fname.lower().endswith(".pdf"):
            text = safe_read_pdf(path)
        if text is None:
            text = safe_read_text(path)

        if not text:
            print(f"[SKIPPED] {fname} (unreadable)\n")
            continue

        text = clean_text(text)
        chunks = chunk_text(text)

        for chunk in chunks:
            kb_entries.append({"source": fname, "content": chunk})

        print(f"   ✔ {len(chunks)} chunks\n")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as out:
        json.dump(kb_entries, out, indent=2)

    print("✅ INGESTION COMPLETE")
    print("📦 Total Chunks:", len(kb_entries))
    print("📁 Output:", OUTPUT_JSON)


if __name__ == "__main__":
    ingest_documents()

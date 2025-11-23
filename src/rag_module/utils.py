import re
import pdfplumber

def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_txt = page.extract_text() or ""
            text += page_txt + "\n"
    return text


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

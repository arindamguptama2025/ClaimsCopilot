import pathlib
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from src.llm import ask

# Multilingual embeddings, so Italian/German questions work too
emb = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2")
db = chromadb.PersistentClient(path="data/chroma")
col = db.get_or_create_collection("policies", embedding_function=emb)

def read_file(p: pathlib.Path):
    if p.suffix.lower() == ".pdf":
        return [(i + 1, pg.extract_text() or "") for i, pg in enumerate(PdfReader(p).pages)]
    return [(1, p.read_text())]

def chunk(text, size=900, overlap=150):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out

def ingest(path) -> int:
    p = pathlib.Path(path)
    ids, docs, metas = [], [], []
    for page, text in read_file(p):
        for n, c in enumerate(chunk(text)):
            ids.append(f"{p.name}-{page}-{n}")
            docs.append(c)
            metas.append({"source": p.name, "page": page})
    if ids:
        col.upsert(ids=ids, documents=docs, metadatas=metas)
    return len(ids)

def retrieve(q: str, k: int = 3):
    r = col.query(query_texts=[q], n_results=k)
    return [{"id": i, "text": d, "meta": m}
            for i, d, m in zip(r["ids"][0], r["documents"][0], r["metadatas"][0])]

def answer(q: str):
    hits = retrieve(q)
    ctx = "\n\n".join(
        f"[{i+1}] ({h['meta']['source']}, p.{h['meta']['page']})\n{h['text']}"
        for i, h in enumerate(hits))
    system = ("You are an insurance policy assistant.\n"
              "Rules:\n"
              "1. Use ONLY the numbered context passages below.\n"
              "2. Cite every claim like [1] or [2].\n"
              "3. If the answer is not in the context, reply exactly: "
              "'I cannot find this in the provided documents.'\n"
              "4. Keep the answer under 120 words. Do not give legal advice.")
    sources = [f"{h['meta']['source']} p.{h['meta']['page']}" for h in hits]
    return ask(system, f"Context:\n{ctx}\n\nQuestion: {q}", module="rag", sources=sources), hits
    system = ("You are an insurance policy assistant.\n"
              "Rules:\n"
              "1. Use ONLY the numbered context passages below.\n"
              "2. Cite every claim like [1] or [2].\n"
              "3. If the answer is not in the context, reply exactly: "
              "'I cannot find this in the provided documents.'\n"
              "4. Keep the answer under 120 words. Do not give legal advice.")
    return ask(system, f"Context:\n{ctx}\n\nQuestion: {q}"), hits

if __name__ == "__main__":
    for f in pathlib.Path("data/policies").glob("*"):
        print(f.name, ingest(f))
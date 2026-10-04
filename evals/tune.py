import json, pathlib
import chromadb
from src.rag import emb, read_file, chunk

qa = [x for x in json.load(open("evals/qa.json", encoding="utf-8")) if x.get("anchor")]
client = chromadb.EphemeralClient()

def build(size, overlap):
    col = client.get_or_create_collection(f"t_{size}_{overlap}", embedding_function=emb)
    ids, docs = [], []
    for p in pathlib.Path("data/policies").glob("*"):
        for page, text in read_file(p):
            for n, c in enumerate(chunk(text, size, overlap)):
                ids.append(f"{p.name}-{page}-{n}")
                docs.append(c)
    col.add(ids=ids, documents=docs)
    return col

print("| chunk | overlap | k=3 | k=5 |\n|---|---|---|---|")
for size, overlap in [(500, 100), (700, 150), (900, 150), (1200, 200)]:
    col = build(size, overlap)
    row = []
    for k in (3, 5):
        hits = sum(any(x["anchor"] in d for d in
                       col.query(query_texts=[x["question"]], n_results=k)["documents"][0])
                   for x in qa)
        row.append(f"{hits / len(qa):.0%}")
    print(f"| {size} | {overlap} | {row[0]} | {row[1]} |")
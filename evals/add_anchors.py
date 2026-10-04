import json
from src.rag import col

qa = json.load(open("evals/qa.json", encoding="utf-8"))
ids = [x["chunk_id"] for x in qa if x.get("chunk_id")]
docs = dict(zip(*[(r) for r in (col.get(ids=ids)["ids"], col.get(ids=ids)["documents"])]))
for x in qa:
    if x.get("chunk_id") in docs:
        d = docs[x["chunk_id"]]
        mid = len(d) // 2
        x["anchor"] = d[mid - 40: mid + 40]
json.dump(qa, open("evals/qa.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(sum("anchor" in x for x in qa), "anchors added")
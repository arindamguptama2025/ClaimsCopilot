import json, random
from src.rag import col
from src.llm import ask

random.seed(1)
data = col.get()   # ids, documents, metadatas of every chunk
order = list(range(len(data["ids"])))
random.shuffle(order)

SYSTEM = ("You write test questions for an insurance policy assistant. Write ONE specific "
          "question that can be answered using only the passage. Do not mention 'the passage'. "
          "Output only the question.")
qa = []
for i in order:
    if len(qa) >= 40:
        break
    doc = data["documents"][i]
    if len(doc) < 500:
        continue
    q = ask(SYSTEM, doc, max_tokens=80, module="eval_gen").strip().strip('"')
    qa.append({"question": q, "chunk_id": data["ids"][i], "source": data["metadatas"][i]["source"]})
    print(len(qa), q)

# Unanswerable questions: the system should refuse these
for q in ["What is the weather in Milan today?", "Is colonizing Mars covered?",
          "Who won the 2022 football World Cup?", "What is the CEO's salary?",
          "Does the policy cover damage caused by dragons?"]:
    qa.append({"question": q, "chunk_id": None, "source": None})

json.dump(qa, open("evals/qa.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
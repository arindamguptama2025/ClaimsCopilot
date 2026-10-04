import os
from src.rag import retrieve
from src.llm import ask

JUDGE = ("You are a strict grader. Given CONTEXT and ANSWER, reply YES only if every factual "
         "claim in the ANSWER is supported by the CONTEXT. Otherwise reply NO. Output one word.")
JUDGE_MODEL = os.getenv("JUDGE_MODEL")

questions = ["Is water damage from a burst pipe covered?",
             "What is the deductible for theft?",
             "How do I cancel the policy?"]
fake = "The policy pays up to EUR 900,000 for this, with no deductible, and covers damage caused by war [1]."

for q in questions:
    ctx = "\n\n".join(h["text"] for h in retrieve(q, k=3))
    v = ask(JUDGE, f"CONTEXT:\n{ctx}\n\nANSWER:\n{fake}", max_tokens=5,
            temperature=0, module="eval_judge", model=JUDGE_MODEL)
    print(q, "->", v.strip())
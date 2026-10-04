import json, os, random, sys
from src.rag import retrieve, answer
from src.llm import ask

N_FAITH = int(sys.argv[1]) if len(sys.argv) > 1 else 15   # faithfulness is slow
JUDGE_MODEL = os.getenv("JUDGE_MODEL")   # e.g. the Phi-4-mini tag

qa = json.load(open("evals/qa.json", encoding="utf-8"))
pos = [x for x in qa if x.get("chunk_id")]       # questions with a known source chunk
neg = [x for x in qa if not x.get("chunk_id")]   # unanswerable questions

def is_hit(x, k=5):
    hits = retrieve(x["question"], k=k)
    if x.get("anchor"):                           # robust to chunk-size changes
        return any(x["anchor"] in h["text"] for h in hits)
    return x["chunk_id"] in [h["id"] for h in hits]

# 1. Retrieval hit-rate@5
hit_n = sum(is_hit(x) for x in pos)
hit_rate = hit_n / len(pos)
print(f"hit-rate@5: {hit_n}/{len(pos)} = {hit_rate:.0%}")

# 2. Faithfulness (LLM-judged), only on questions the system actually answered
JUDGE = ("You are a strict grader. Given CONTEXT and ANSWER, reply YES only if every factual "
         "claim in the ANSWER is supported by the CONTEXT. Otherwise reply NO. Output one word.")
random.seed(0)
sample = random.sample(pos, min(N_FAITH, len(pos)))
faithful = answered = 0
for x in sample:
    a, h = answer(x["question"])
    if "cannot find" in a.lower():
        continue                                  # abstentions are not faithfulness evidence
    answered += 1
    ctx = "\n\n".join(c["text"] for c in h)
    v = ask(JUDGE, f"CONTEXT:\n{ctx}\n\nANSWER:\n{a}", max_tokens=5, temperature=0,
            module="eval_judge", model=JUDGE_MODEL)
    faithful += v.strip().upper().startswith("YES")
faith = faithful / max(answered, 1)
print(f"faithfulness: {faithful}/{answered} answered (of {len(sample)} sampled) = {faith:.0%}")

# 3. Correct refusal on unanswerable questions (prints any failures)
refused = 0
for x in neg:
    a, _ = answer(x["question"])
    ok = "cannot find" in a.lower()
    refused += ok
    if not ok:
        print("DID NOT REFUSE:", x["question"], "->", a[:150])
print(f"correct refusals: {refused}/{len(neg)}")

res = {"hit_rate_at_5": hit_rate, "faithfulness": faith, "refusal_rate": refused / len(neg),
       "n_retrieval": len(pos), "n_faithfulness": answered, "n_refusal": len(neg)}
json.dump(res, open("evals/results.json", "w"), indent=2)

print("\n| Metric | Result |\n|---|---|")
print(f"| Retrieval hit-rate@5 ({len(pos)} Qs) | {hit_rate:.0%} |")
print(f"| Faithfulness ({answered} answered Qs, LLM-judged) | {faith:.0%} |")
print(f"| Correct refusals ({len(neg)} Qs) | {refused}/{len(neg)} |")
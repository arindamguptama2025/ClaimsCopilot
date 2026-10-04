import pathlib
from src.llm import ask

TYPES = ["Home insurance", "Auto insurance", "Travel insurance", "Health insurance"]
OUT = pathlib.Path("data/policies")
OUT.mkdir(parents=True, exist_ok=True)

SYSTEM = ("You write realistic but fictional insurance policy wordings for a fictional "
          "insurer called 'Lombarda Assicurazioni'. Use numbered clauses (e.g. 'Clause 4.2'), "
          "with sections: Definitions, Covered Events, Exclusions, Limits and Deductibles, "
          "Claims Procedure, Cancellation. Include specific figures and edge cases "
          "(e.g. gradual vs sudden water damage). About 1200 words.")

for t in TYPES:
    text = ask(SYSTEM, f"Write the full policy wording for: {t}", max_tokens=3000)
    (OUT / f"{t.lower().replace(' ', '_')}.txt").write_text(text)
    print("created", t)
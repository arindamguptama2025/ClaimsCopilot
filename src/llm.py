import os
import ollama
from src.guardrails import mask_pii, check_request
from src.audit import log

MODEL = os.getenv("LOCAL_MODEL", "gemma3:4b-it-q4_K_M")   # your tag
REFUSAL = ("I can't help with that request. I can answer questions about the "
           "policy documents and claims data.")

def ask(system: str, user: str, max_tokens: int = 1000, temperature: float = 0.2,
        json_mode: bool = False, module: str = "general", sources=None, model=None) -> str:
    safe_user, found = mask_pii(user)          # PII never reaches the model or the log
    blocked = check_request(user)
    if blocked:
        log(module, safe_user, REFUSAL, sources, found, blocked)
        return REFUSAL
    r = ollama.chat(
        model=model or MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": safe_user}],
        format="json" if json_mode else None,
        options={"num_predict": max_tokens, "temperature": temperature, "num_ctx": 8192},
    )
    out = r["message"]["content"]
    log(module, safe_user, out, sources, found)
    return out
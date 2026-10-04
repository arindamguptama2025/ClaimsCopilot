import re

# Order matters: specific patterns first, generic phone last. Demo-grade, not production-grade.
PATTERNS = [
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,4})?\b")),
    ("FISCAL_CODE", re.compile(r"\b[A-Z]{6}\d{2}[A-EHLMPR-T]\d{2}[A-Z]\d{3}[A-Z]\b", re.I)),
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+|00)\d[\d\s().-]{7,}\d|\b3\d{2}[\s.-]?\d{6,7}\b")),
]

REFUSE = [
    (re.compile(r"ignore (all |any |previous |the |your )*instructions", re.I), "prompt_injection"),
    (re.compile(r"(reveal|show|print).{0,30}system prompt", re.I), "prompt_injection"),
    (re.compile(r"\bjailbreak\b", re.I), "prompt_injection"),
    (re.compile(r"\b(drop|delete)\s+table\b", re.I), "destructive_sql"),
]

def mask_pii(text: str):
    found = []
    for name, rx in PATTERNS:
        text, n = rx.subn(f"[{name}]", text)
        if n:
            found.append(name)
    return text, found

def check_request(text: str):
    """Returns a reason string if the request must be refused, else None."""
    for rx, reason in REFUSE:
        if rx.search(text):
            return reason
    return None
import re, sqlite3
from datetime import date
import pandas as pd
from src.llm import ask

DB = "data/claims.db"

SYSTEM = """You convert questions into ONE SQLite SELECT query. Output only the SQL, no explanation.
Today's date is {today}.

Schema:
claims(id INTEGER, claim_type TEXT ['Auto','Home','Travel','Health'],
       region TEXT ['Lombardy','Lazio','Veneto','Piedmont','Tuscany','Campania'],
       amount REAL (EUR), status TEXT ['Open','In Review','Approved','Rejected','Paid'],
       incident_date TEXT (YYYY-MM-DD))

Examples:
Q: Show open auto claims over 10000 in Lombardy
SQL: SELECT * FROM claims WHERE claim_type='Auto' AND status='Open' AND amount>10000 AND region='Lombardy'

Q: Total claim amount by region
SQL: SELECT region, SUM(amount) AS total_amount FROM claims GROUP BY region ORDER BY total_amount DESC

Q: How many claims were rejected last month
SQL: SELECT COUNT(*) AS n FROM claims WHERE status='Rejected' AND incident_date >= date('now','start of month','-1 month') AND incident_date < date('now','start of month')

Q: Average amount of home claims this quarter
SQL: SELECT AVG(amount) AS avg_amount FROM claims WHERE claim_type='Home' AND incident_date >= date('now','start of month','-' || ((CAST(strftime('%m','now') AS INT)-1)%3) || ' months')
"""

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|pragma|replace|truncate|vacuum)\b", re.I)

def clean(raw: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", raw, re.S | re.I)
    return (m.group(1) if m else raw).strip().rstrip(";").strip()

def is_safe(sql: str) -> bool:
    return (sql.lower().startswith("select") and ";" not in sql
            and "--" not in sql and "/*" not in sql and not FORBIDDEN.search(sql))

def nl_to_sql(question: str):
    """Returns (sql, dataframe, error)."""
    raw = ask(SYSTEM.format(today=date.today().isoformat()), question,
              max_tokens=300, temperature=0, module="nl2sql")
    sql = clean(raw)
    if not sql.lower().startswith("select"):
        return None, None, f"Could not generate a SELECT query. Model said: {raw[:200]}"
    if not is_safe(sql):
        return sql, None, "Blocked: only a single read-only SELECT query is allowed."
    try:
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)   # read-only connection
        df = pd.read_sql_query(f"SELECT * FROM ({sql}) LIMIT 200", con)
        con.close()
        return sql, df, None
    except Exception as e:
        return sql, None, f"Query failed: {e}"
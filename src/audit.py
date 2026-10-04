import sqlite3, json, pathlib
from datetime import datetime, timezone
import pandas as pd

DB = "data/audit.db"

def _con():
    pathlib.Path("data").mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, module TEXT, prompt TEXT,
        answer TEXT, sources TEXT, pii_found TEXT, blocked TEXT)""")
    return con

def log(module, prompt, answer, sources=None, pii_found=None, blocked=None):
    con = _con()
    con.execute(
        "INSERT INTO audit_log (ts,module,prompt,answer,sources,pii_found,blocked) VALUES (?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"), module, prompt, answer,
         json.dumps(sources or []), ",".join(pii_found or []), blocked))
    con.commit(); con.close()

def recent(n: int = 50):
    con = _con()
    df = pd.read_sql_query(
        "SELECT ts,module,prompt,answer,sources,pii_found,blocked "
        "FROM audit_log ORDER BY id DESC LIMIT ?", con, params=(n,))
    con.close()
    return df
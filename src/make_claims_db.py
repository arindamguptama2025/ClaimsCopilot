import sqlite3, random, pathlib
from datetime import date, timedelta

random.seed(42)
TYPES = ["Auto", "Home", "Travel", "Health"]
REGIONS = ["Lombardy", "Lazio", "Veneto", "Piedmont", "Tuscany", "Campania"]
STATUS = ["Open", "In Review", "Approved", "Rejected", "Paid"]
BASE = {"Auto": 8.2, "Home": 8.3, "Travel": 6.5, "Health": 7.2}  # controls typical amounts

pathlib.Path("data").mkdir(exist_ok=True)
con = sqlite3.connect("data/claims.db")
con.execute("DROP TABLE IF EXISTS claims")
con.execute("""CREATE TABLE claims (
    id INTEGER PRIMARY KEY, claim_type TEXT, region TEXT,
    amount REAL, status TEXT, incident_date TEXT)""")

today = date.today()
rows = []
for i in range(1, 501):
    t = random.choice(TYPES)
    rows.append((
        i, t, random.choice(REGIONS),
        round(random.lognormvariate(BASE[t], 1.1), 2),
        random.choices(STATUS, weights=[25, 20, 20, 10, 25])[0],
        (today - timedelta(days=random.randint(0, 364))).isoformat(),
    ))
con.executemany("INSERT INTO claims VALUES (?,?,?,?,?,?)", rows)
con.commit(); con.close()
print("created 500 claims")
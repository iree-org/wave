import sqlite3
import pandas as pd
from sys import argv

conn = sqlite3.connect(argv[1])
df_top = pd.read_sql_query("SELECT * from top", conn)
df_busy = pd.read_sql_query("SELECT * from busy", conn)
conn.close()
print(df_top.head(100))
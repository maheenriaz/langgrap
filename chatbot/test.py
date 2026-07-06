import sqlite3

conn = sqlite3.connect("chatbot.db")

conn.execute("""
CREATE TABLE IF NOT EXISTS test (
    id INTEGER
)
""")

conn.commit()

print("OK")
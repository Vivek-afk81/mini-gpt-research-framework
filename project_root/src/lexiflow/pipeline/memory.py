"""Small SQLite-backed profile and request memory."""

import json
import sqlite3
import time


class Memory:
    def __init__(self, path="lexiflow.db"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users(
                user_id TEXT PRIMARY KEY, name TEXT, roll_no TEXT,
                section TEXT, subjects TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS requests(
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
                ts REAL NOT NULL, hinglish TEXT, english TEXT,
                category TEXT, tone TEXT, draft TEXT
            );
            """
        )
        self.db.commit()

    def upsert_profile(self, user_id, name="", roll_no="", section="", subjects=()):
        self.db.execute(
            """INSERT INTO users VALUES(?,?,?,?,?)
               ON CONFLICT(user_id) DO UPDATE SET name=excluded.name,
               roll_no=excluded.roll_no, section=excluded.section,
               subjects=excluded.subjects""",
            (user_id, name, roll_no, section, json.dumps(list(subjects))),
        )
        self.db.commit()

    def profile(self, user_id):
        row = self.db.execute(
            "SELECT name, roll_no, section, subjects FROM users WHERE user_id=?", (user_id,)
        ).fetchone()
        if not row:
            return {}
        return {"name": row[0], "roll_no": row[1], "section": row[2], "subjects": json.loads(row[3])}

    def log_request(self, user_id, hinglish, english, category, tone, draft):
        self.db.execute(
            "INSERT INTO requests(user_id,ts,hinglish,english,category,tone,draft) VALUES(?,?,?,?,?,?,?)",
            (user_id, time.time(), hinglish, english, category, tone, draft),
        )
        self.db.commit()

    def recent(self, user_id, category=None, limit=3):
        query = "SELECT ts, english, category FROM requests WHERE user_id=?"
        args = [user_id]
        if category:
            query += " AND category=?"
            args.append(category)
        query += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        return self.db.execute(query, args).fetchall()

    def close(self):
        self.db.close()
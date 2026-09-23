"""SQLite 持久化层：对话会话、消息历史、接口调用统计。"""
import os
import sqlite3
from datetime import datetime
from typing import Dict, List

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "chat.db")


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = _conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS sessions (
        session_id    TEXT PRIMARY KEY,
        title         TEXT,
        message_count INTEGER DEFAULT 0,
        created_at    TEXT,
        updated_at    TEXT
    );
    CREATE TABLE IF NOT EXISTS messages (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  TEXT,
        role        TEXT,
        content     TEXT,
        created_at  TEXT
    );
    CREATE TABLE IF NOT EXISTS api_stats (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        endpoint    TEXT,
        success     INTEGER,
        duration_ms INTEGER,
        created_at  TEXT
    );
    """)
    conn.commit()
    conn.close()


# ---------- 对话历史 ----------
def get_history(session_id: str) -> List[Dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id",
        (session_id,)).fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def append_message(session_id: str, role: str, content: str):
    conn = _conn()
    now = datetime.now().isoformat()
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
        (session_id, role, content, now))
    cnt = conn.execute(
        "SELECT COUNT(*) c FROM messages WHERE session_id=?", (session_id,)).fetchone()["c"]
    conn.execute("""
        INSERT INTO sessions (session_id, title, message_count, created_at, updated_at)
        VALUES (?,?,0,?,?)
        ON CONFLICT(session_id) DO UPDATE SET message_count=excluded.message_count, updated_at=excluded.updated_at
    """, (session_id, "新对话", now, now))
    conn.commit()
    conn.close()


def touch_session(session_id: str, title: str = None):
    conn = _conn()
    now = datetime.now().isoformat()
    row = conn.execute("SELECT session_id FROM sessions WHERE session_id=?",
                       (session_id,)).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO sessions (session_id, title, message_count, created_at, updated_at) VALUES (?,?,0,?,?)",
            (session_id, title or "新对话", now, now))
    else:
        conn.execute("UPDATE sessions SET updated_at=? WHERE session_id=?",
                     (now, session_id))
    conn.commit()
    conn.close()


def list_sessions() -> List[Dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT session_id, title, message_count, created_at, updated_at "
        "FROM sessions ORDER BY updated_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def clear_messages(session_id: str):
    """清空某会话的消息记录，但保留会话条目。"""
    conn = _conn()
    conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
    conn.execute("UPDATE sessions SET message_count=0 WHERE session_id=?", (session_id,))
    conn.commit()
    conn.close()


def delete_session(session_id: str):
    conn = _conn()
    conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
    conn.commit()
    conn.close()


# ---------- 接口调用统计 ----------
def record_call(endpoint: str, success: bool, duration_ms: int):
    """记录一次接口调用。"""
    conn = _conn()
    conn.execute(
        "INSERT INTO api_stats (endpoint, success, duration_ms, created_at) VALUES (?,?,?,?)",
        (endpoint, 1 if success else 0, int(duration_ms), datetime.now().isoformat()))
    conn.commit()
    conn.close()


def stats_overview() -> Dict:
    """汇总各接口调用次数与失败率。"""
    conn = _conn()
    rows = conn.execute("""
        SELECT endpoint,
               COUNT(*) AS total,
               SUM(success) AS ok,
               ROUND(100.0*(COUNT(*)-SUM(success))/COUNT(*), 1) AS fail_rate,
               AVG(duration_ms) AS avg_ms
        FROM api_stats GROUP BY endpoint ORDER BY total DESC
    """).fetchall()
    total = conn.execute("SELECT COUNT(*) c FROM api_stats").fetchone()["c"]
    failed = conn.execute("SELECT COUNT(*) c FROM api_stats WHERE success=0").fetchone()["c"]
    conn.close()
    items = []
    for r in rows:
        items.append({
            "接口": r["endpoint"],
            "调用次数": r["total"],
            "成功次数": r["ok"],
            "失败率%": r["fail_rate"] or 0.0,
            "平均耗时ms": round(r["avg_ms"] or 0, 1),
        })
    return {
        "总调用": total,
        "总失败": failed,
        "整体失败率%": round(100.0 * failed / total, 1) if total else 0.0,
        "各接口": items,
    }

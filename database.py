"""
SilverVault AI - Database Module
SQLite database management with WAL mode, foreign keys, and comprehensive schemas
for users, documents, chunks, chat messages, MCQ tests, continuous viva sessions,
and topic-level student learning analysis reports.
"""

import os
import sqlite3
import hashlib
import hmac
import secrets
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "silvervault.db")
os.makedirs(DATA_DIR, exist_ok=True)

def connect() -> sqlite3.Connection:
    """Connect to SQLite database with WAL mode and foreign keys enabled."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def _hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 240_000)
    return salt.hex(), digest.hex()

def init_db():
    """Initialize all database tables, indexes, and constraints."""
    with connect() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT NOT NULL,
            email TEXT NOT NULL DEFAULT '',
            department TEXT NOT NULL DEFAULT 'Computer Science & Engineering',
            default_semester TEXT NOT NULL DEFAULT 'Semester 1',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            semester TEXT NOT NULL,
            subject TEXT NOT NULL,
            title TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL DEFAULT '',
            chunk_count INTEGER NOT NULL DEFAULT 0,
            page_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            page_number INTEGER NOT NULL DEFAULT 1,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            semester TEXT NOT NULL,
            subject TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            sources_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            semester TEXT NOT NULL,
            subject TEXT NOT NULL,
            test_type TEXT NOT NULL, -- 'MCQ' or 'Verbal viva'
            score REAL NOT NULL,
            total REAL NOT NULL,
            percentage REAL NOT NULL,
            difficulty TEXT NOT NULL DEFAULT 'Standard',
            duration_seconds INTEGER NOT NULL DEFAULT 0,
            review_json TEXT NOT NULL DEFAULT '[]',
            topic_breakdown_json TEXT NOT NULL DEFAULT '{}',
            weak_topics_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS viva_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            semester TEXT NOT NULL,
            subject TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active', -- 'active' or 'completed'
            question_count INTEGER NOT NULL DEFAULT 0,
            total_score REAL NOT NULL DEFAULT 0,
            total_possible REAL NOT NULL DEFAULT 0,
            current_topic TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            completed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS viva_turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES viva_sessions(id) ON DELETE CASCADE,
            turn_number INTEGER NOT NULL,
            topic TEXT NOT NULL DEFAULT 'General',
            question TEXT NOT NULL,
            context TEXT NOT NULL DEFAULT '',
            student_answer TEXT NOT NULL DEFAULT '',
            feedback TEXT NOT NULL DEFAULT '',
            score REAL NOT NULL DEFAULT 0,
            missing_concepts_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS learning_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            semester TEXT NOT NULL,
            subject TEXT NOT NULL,
            test_type TEXT NOT NULL, -- 'MCQ' or 'Verbal viva'
            title TEXT NOT NULL,
            report_markdown TEXT NOT NULL,
            mastery_level TEXT NOT NULL DEFAULT 'Developing', -- 'Mastered', 'Developing', 'Needs Attention'
            percentage REAL NOT NULL DEFAULT 0,
            weak_topics_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS student_topic_mastery (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            total_tested INTEGER NOT NULL DEFAULT 0,
            correct_count INTEGER NOT NULL DEFAULT 0,
            mastery_score REAL NOT NULL DEFAULT 0.5, -- 0.0 to 1.0 (Bayesian Knowledge State)
            last_updated TEXT NOT NULL,
            UNIQUE(user_id, subject, topic)
        );

        CREATE INDEX IF NOT EXISTS idx_docs_owner ON documents(owner_id, semester, subject);
        CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id, chunk_index);
        CREATE INDEX IF NOT EXISTS idx_chat_user_sub ON chat_messages(user_id, semester, subject);
        CREATE INDEX IF NOT EXISTS idx_tests_user ON test_results(user_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_viva_session_user ON viva_sessions(user_id, status);
        CREATE INDEX IF NOT EXISTS idx_viva_turns_session ON viva_turns(session_id, turn_number);
        CREATE INDEX IF NOT EXISTS idx_reports_user ON learning_reports(user_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_mastery_user_sub ON student_topic_mastery(user_id, subject);
        """)

# --- User Management ---

def register_user(username: str, password: str, display_name: str, department: str = "Computer Science & Engineering") -> Tuple[bool, str]:
    if len(username.strip()) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    salt, hashed = _hash_password(password)
    try:
        with connect() as c:
            c.execute("""
                INSERT INTO users(username, password_salt, password_hash, display_name, department, created_at)
                VALUES(?, ?, ?, ?, ?, ?)
            """, (username.strip().lower(), salt, hashed, display_name.strip() or username.strip(), department, datetime.now().isoformat(timespec="seconds")))
        return True, "Account created successfully!"
    except sqlite3.IntegrityError:
        return False, "That username is already taken. Please choose another."

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    with connect() as c:
        row = c.execute("SELECT * FROM users WHERE username = ?", (username.strip().lower(),)).fetchone()
    if not row:
        return None
    salt = bytes.fromhex(row["password_salt"])
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 240_000).hex()
    if not hmac.compare_digest(digest, row["password_hash"]):
        return None
    return dict(row)

def get_user(user_id: int) -> Optional[Dict[str, Any]]:
    with connect() as c:
        row = c.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None

def update_profile(user_id: int, display_name: str, email: str, department: str, default_semester: str):
    with connect() as c:
        c.execute("""
            UPDATE users SET display_name = ?, email = ?, department = ?, default_semester = ?
            WHERE id = ?
        """, (display_name, email, department, default_semester, user_id))

# --- Document & Chunk Management ---

def save_document(owner_id: Optional[int], semester: str, subject: str, title: str, filename: str, file_path: str, chunk_count: int, page_count: int) -> int:
    with connect() as c:
        cur = c.execute("""
            INSERT INTO documents(owner_id, semester, subject, title, filename, file_path, chunk_count, page_count, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (owner_id, semester, subject, title, filename, file_path, chunk_count, page_count, datetime.now().isoformat(timespec="seconds")))
        return cur.lastrowid

def list_documents(user_id: int, include_shared: bool = True) -> List[Dict[str, Any]]:
    with connect() as c:
        if include_shared:
            rows = c.execute("""
                SELECT * FROM documents
                WHERE owner_id = ? OR owner_id IS NULL
                ORDER BY semester, subject, title
            """, (user_id,)).fetchall()
        else:
            rows = c.execute("""
                SELECT * FROM documents
                WHERE owner_id = ?
                ORDER BY created_at DESC
            """, (user_id,)).fetchall()
    return [dict(r) for r in rows]

def get_document(document_id: int) -> Optional[Dict[str, Any]]:
    with connect() as c:
        row = c.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    return dict(row) if row else None

def delete_document(document_id: int, user_id: int):
    with connect() as c:
        c.execute("DELETE FROM documents WHERE id = ? AND (owner_id = ? OR owner_id IS NULL)", (document_id, user_id))
        c.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

def save_chunks_with_pages(document_id: int, chunks_data: List[Dict[str, Any]]):
    """
    Save chunk list where each item has {"chunk_index": int, "page_number": int, "content": str, "token_count": int}.
    """
    with connect() as c:
        c.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        c.executemany("""
            INSERT INTO chunks(document_id, chunk_index, page_number, content, token_count)
            VALUES(?, ?, ?, ?, ?)
        """, [(document_id, ch["chunk_index"], ch.get("page_number", 1), ch["content"], ch.get("token_count", 0)) for ch in chunks_data])

def get_chunks_for_docs(document_ids: List[int], user_id: int) -> List[Dict[str, Any]]:
    if not document_ids:
        return []
    placeholders = ",".join("?" for _ in document_ids)
    params = list(document_ids) + [user_id]
    with connect() as c:
        rows = c.execute(f"""
            SELECT ch.id, ch.document_id, ch.chunk_index, ch.page_number, ch.content, ch.token_count,
                   d.title, d.subject, d.semester, d.owner_id
            FROM chunks ch
            JOIN documents d ON d.id = ch.document_id
            WHERE ch.document_id IN ({placeholders})
              AND (d.owner_id = ? OR d.owner_id IS NULL)
            ORDER BY ch.document_id, ch.chunk_index
        """, params).fetchall()
    return [dict(r) for r in rows]

# --- Chat Messages ---

def save_chat_message(user_id: int, semester: str, subject: str, role: str, content: str, sources: List[str] = None):
    sources_json = json.dumps(sources or [], ensure_ascii=False)
    with connect() as c:
        c.execute("""
            INSERT INTO chat_messages(user_id, semester, subject, role, content, sources_json, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
        """, (user_id, semester, subject, role, content, sources_json, datetime.now().isoformat(timespec="seconds")))

def get_chat_history(user_id: int, semester: str, subject: str, limit: int = 100) -> List[Dict[str, Any]]:
    with connect() as c:
        rows = c.execute("""
            SELECT role, content, sources_json, created_at
            FROM chat_messages
            WHERE user_id = ? AND semester = ? AND subject = ?
            ORDER BY id ASC
            LIMIT ?
        """, (user_id, semester, subject, limit)).fetchall()
    res = []
    for r in rows:
        d = dict(r)
        d["sources"] = json.loads(d.get("sources_json", "[]"))
        res.append(d)
    return res

def clear_chat_history(user_id: int, semester: str, subject: str):
    with connect() as c:
        c.execute("DELETE FROM chat_messages WHERE user_id = ? AND semester = ? AND subject = ?", (user_id, semester, subject))

# --- Test Results & Analytics ---

def save_test_result(
    user_id: int, semester: str, subject: str, test_type: str, score: float, total: float,
    percentage: float, difficulty: str = "Standard", duration_seconds: int = 0,
    review_list: List[Dict[str, Any]] = None, topic_breakdown: Dict[str, Any] = None,
    weak_topics: List[str] = None
) -> int:
    with connect() as c:
        cur = c.execute("""
            INSERT INTO test_results(
                user_id, semester, subject, test_type, score, total, percentage,
                difficulty, duration_seconds, review_json, topic_breakdown_json, weak_topics_json, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, semester, subject, test_type, score, total, percentage,
            difficulty, duration_seconds,
            json.dumps(review_list or [], ensure_ascii=False),
            json.dumps(topic_breakdown or {}, ensure_ascii=False),
            json.dumps(weak_topics or [], ensure_ascii=False),
            datetime.now().isoformat(timespec="seconds")
        ))
        return cur.lastrowid

def get_test_results(user_id: int, limit: int = 100) -> List[Dict[str, Any]]:
    with connect() as c:
        rows = c.execute("""
            SELECT * FROM test_results
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (user_id, limit)).fetchall()
    res = []
    for r in rows:
        d = dict(r)
        d["review"] = json.loads(d.get("review_json", "[]"))
        d["topic_breakdown"] = json.loads(d.get("topic_breakdown_json", "{}"))
        d["weak_topics"] = json.loads(d.get("weak_topics_json", "[]"))
        res.append(d)
    return res

# --- Continuous Viva Sessions ---

def create_viva_session(user_id: int, semester: str, subject: str) -> int:
    with connect() as c:
        cur = c.execute("""
            INSERT INTO viva_sessions(user_id, semester, subject, status, created_at)
            VALUES(?, ?, ?, 'active', ?)
        """, (user_id, semester, subject, datetime.now().isoformat(timespec="seconds")))
        return cur.lastrowid

def get_viva_session(session_id: int, user_id: int) -> Optional[Dict[str, Any]]:
    with connect() as c:
        row = c.execute("""
            SELECT * FROM viva_sessions WHERE id = ? AND user_id = ?
        """, (session_id, user_id)).fetchone()
    return dict(row) if row else None

def save_viva_turn(
    session_id: int, turn_number: int, topic: str, question: str, context: str,
    student_answer: str, feedback: str, score: float, missing_concepts: List[str] = None
):
    with connect() as c:
        c.execute("""
            INSERT INTO viva_turns(
                session_id, turn_number, topic, question, context, student_answer,
                feedback, score, missing_concepts_json, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, turn_number, topic, question, context, student_answer,
            feedback, score, json.dumps(missing_concepts or [], ensure_ascii=False),
            datetime.now().isoformat(timespec="seconds")
        ))
        c.execute("""
            UPDATE viva_sessions
            SET question_count = question_count + 1,
                total_score = total_score + ?,
                total_possible = total_possible + 10,
                current_topic = ?
            WHERE id = ?
        """, (score, topic, session_id))

def get_viva_turns(session_id: int) -> List[Dict[str, Any]]:
    with connect() as c:
        rows = c.execute("""
            SELECT * FROM viva_turns WHERE session_id = ? ORDER BY turn_number ASC
        """, (session_id,)).fetchall()
    res = []
    for r in rows:
        d = dict(r)
        d["missing_concepts"] = json.loads(d.get("missing_concepts_json", "[]"))
        res.append(d)
    return res

def complete_viva_session(session_id: int, user_id: int):
    with connect() as c:
        c.execute("""
            UPDATE viva_sessions
            SET status = 'completed', completed_at = ?
            WHERE id = ? AND user_id = ?
        """, (datetime.now().isoformat(timespec="seconds"), session_id, user_id))

def get_viva_history(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    with connect() as c:
        rows = c.execute("""
            SELECT * FROM viva_sessions
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (user_id, limit)).fetchall()
    return [dict(r) for r in rows]

# --- Learning Reports & Topic Mastery ---

def save_learning_report(
    user_id: int, semester: str, subject: str, test_type: str, title: str,
    report_markdown: str, mastery_level: str = "Developing", percentage: float = 0.0,
    weak_topics: List[str] = None
) -> int:
    with connect() as c:
        cur = c.execute("""
            INSERT INTO learning_reports(
                user_id, semester, subject, test_type, title, report_markdown,
                mastery_level, percentage, weak_topics_json, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, semester, subject, test_type, title, report_markdown,
            mastery_level, percentage, json.dumps(weak_topics or [], ensure_ascii=False),
            datetime.now().isoformat(timespec="seconds")
        ))
        return cur.lastrowid

def get_learning_reports(user_id: int, limit: int = 100) -> List[Dict[str, Any]]:
    with connect() as c:
        rows = c.execute("""
            SELECT * FROM learning_reports
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (user_id, limit)).fetchall()
    res = []
    for r in rows:
        d = dict(r)
        d["weak_topics"] = json.loads(d.get("weak_topics_json", "[]"))
        res.append(d)
    return res

def delete_learning_report(report_id: int, user_id: int):
    with connect() as c:
        c.execute("DELETE FROM learning_reports WHERE id = ? AND user_id = ?", (report_id, user_id))

def update_topic_mastery(user_id: int, subject: str, topic: str, is_correct: bool):
    """
    Update topic mastery state using an exponential moving average (Knowledge Tracing).
    """
    now = datetime.now().isoformat(timespec="seconds")
    with connect() as c:
        row = c.execute("""
            SELECT * FROM student_topic_mastery
            WHERE user_id = ? AND subject = ? AND topic = ?
        """, (user_id, subject, topic)).fetchone()

        if row:
            total = row["total_tested"] + 1
            correct = row["correct_count"] + (1 if is_correct else 0)
            # Alpha factor 0.3 for EMA
            obs = 1.0 if is_correct else 0.0
            new_mastery = round(0.7 * row["mastery_score"] + 0.3 * obs, 3)
            c.execute("""
                UPDATE student_topic_mastery
                SET total_tested = ?, correct_count = ?, mastery_score = ?, last_updated = ?
                WHERE id = ?
            """, (total, correct, new_mastery, now, row["id"]))
        else:
            initial_mastery = 0.8 if is_correct else 0.2
            c.execute("""
                INSERT INTO student_topic_mastery(user_id, subject, topic, total_tested, correct_count, mastery_score, last_updated)
                VALUES(?, ?, ?, 1, ?, ?, ?)
            """, (user_id, subject, topic, 1 if is_correct else 0, initial_mastery, now))

def get_student_topic_mastery(user_id: int, subject: Optional[str] = None) -> List[Dict[str, Any]]:
    with connect() as c:
        if subject:
            rows = c.execute("""
                SELECT * FROM student_topic_mastery
                WHERE user_id = ? AND subject = ?
                ORDER BY mastery_score ASC, total_tested DESC
            """, (user_id, subject)).fetchall()
        else:
            rows = c.execute("""
                SELECT * FROM student_topic_mastery
                WHERE user_id = ?
                ORDER BY mastery_score ASC, total_tested DESC
            """, (user_id,)).fetchall()
    return [dict(r) for r in rows]

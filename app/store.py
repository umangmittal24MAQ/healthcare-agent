from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from app.config import get_settings


def _connect() -> sqlite3.Connection:
    path = Path(get_settings().database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                model TEXT NOT NULL,
                safety_status TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                comment TEXT,
                reviewed_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                detail_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )


def save_run(run_id: str, created_at: str, model: str, safety_status: str, payload: dict[str, Any]) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO runs(run_id, created_at, model, safety_status, payload_json) VALUES (?, ?, ?, ?, ?)",
            (run_id, created_at, model, safety_status, json.dumps(payload, default=str)),
        )
        conn.execute(
            "INSERT INTO audit_events(run_id, event_type, detail_json, created_at) VALUES (?, ?, ?, ?)",
            (run_id, "pipeline_completed", json.dumps({"safety_status": safety_status}), created_at),
        )


def get_run(run_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT payload_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return json.loads(row["payload_json"]) if row else None


def save_review(run_id: str, decision: str, comment: str | None, reviewed_at: str) -> None:
    with _connect() as conn:
        exists = conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if not exists:
            raise KeyError(run_id)
        conn.execute(
            "INSERT INTO reviews(run_id, decision, comment, reviewed_at) VALUES (?, ?, ?, ?)",
            (run_id, decision, comment, reviewed_at),
        )
        conn.execute(
            "INSERT INTO audit_events(run_id, event_type, detail_json, created_at) VALUES (?, ?, ?, ?)",
            (run_id, "clinician_review", json.dumps({"decision": decision, "comment": comment}), reviewed_at),
        )

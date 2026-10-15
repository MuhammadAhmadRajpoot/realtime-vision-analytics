"""SQLite persistence for analytics events."""

from __future__ import annotations

import csv
import io
import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rva.analytics.types import Event

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    ts          TEXT    NOT NULL,
    video_t     REAL    NOT NULL,
    type        TEXT    NOT NULL,
    name        TEXT    NOT NULL,
    direction   TEXT,
    track_id    INTEGER NOT NULL,
    class_name  TEXT    NOT NULL,
    dwell_s     REAL
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
"""

COLUMNS = ("id", "session_id", "ts", "video_t", "type", "name", "direction", "track_id", "class_name", "dwell_s")


class EventStore:
    """Thread-safe event log. One connection, serialised with a lock (writes are tiny)."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")

    def add(self, session_id: str, events: Iterable[Event]) -> int:
        rows = [
            (session_id, e.ts, e.video_t, str(e.type), e.name, e.direction, e.track_id, e.class_name, e.dwell_s)
            for e in events
        ]
        if not rows:
            return 0
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT INTO events (session_id, ts, video_t, type, name, direction, track_id, class_name, dwell_s)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def query(
        self,
        limit: int = 100,
        event_type: str | None = None,
        session_id: str | None = None,
        after_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Most recent events first."""
        where, params = self._filters(event_type, session_id, after_id)
        sql = f"SELECT {', '.join(COLUMNS)} FROM events{where} ORDER BY id DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(sql, [*params, limit]).fetchall()
        return [dict(r) for r in rows]

    def count(self, event_type: str | None = None, session_id: str | None = None) -> int:
        where, params = self._filters(event_type, session_id, None)
        with self._lock:
            return int(self._conn.execute(f"SELECT COUNT(*) FROM events{where}", params).fetchone()[0])

    def export_csv(self, event_type: str | None = None, session_id: str | None = None) -> str:
        where, params = self._filters(event_type, session_id, None)
        with self._lock:
            rows = self._conn.execute(f"SELECT {', '.join(COLUMNS)} FROM events{where} ORDER BY id", params)
            buf = io.StringIO()
            writer = csv.writer(buf, lineterminator="\n")
            writer.writerow(COLUMNS)
            writer.writerows(tuple(r) for r in rows)
        return buf.getvalue()

    @staticmethod
    def _filters(event_type: str | None, session_id: str | None, after_id: int | None) -> tuple[str, list[Any]]:
        clauses, params = [], []
        if event_type:
            clauses.append("type = ?")
            params.append(event_type)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if after_id is not None:
            clauses.append("id > ?")
            params.append(after_id)
        return (" WHERE " + " AND ".join(clauses) if clauses else ""), params

    def close(self) -> None:
        with self._lock:
            self._conn.close()

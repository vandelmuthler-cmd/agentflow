from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


class SQLiteCheckpointBackend:
    """Own the SQLite connection used by LangGraph's synchronous checkpointer."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(database_path), check_same_thread=False)
        self.saver = SqliteSaver(self.connection)
        self.saver.setup()
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self.connection.close()
        self._closed = True

    def __enter__(self) -> "SQLiteCheckpointBackend":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

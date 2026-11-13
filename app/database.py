import json
import sqlite3
from datetime import datetime
from pathlib import Path


# Database duoc luu trong thu muc app
DATABASE_PATH = Path(__file__).resolve().parent / "ckd_history.db"


def get_connection():
    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=10
    )
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    """Tao bang lich su neu chua ton tai."""

    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                prediction_label TEXT NOT NULL,
                ckd_probability REAL NOT NULL,
                not_ckd_probability REAL NOT NULL,
                model_name TEXT NOT NULL,
                input_data TEXT NOT NULL
            )
        """)

        connection.commit()


def save_history(
    prediction,
    prediction_label,
    ckd_probability,
    not_ckd_probability,
    model_name,
    input_data
):
    """Luu mot lan phan tich va tra ve ma lich su."""

    created_at = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    input_json = json.dumps(
        input_data,
        ensure_ascii=False,
        allow_nan=False
    )

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO prediction_history (
                created_at,
                prediction,
                prediction_label,
                ckd_probability,
                not_ckd_probability,
                model_name,
                input_data
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                int(prediction),
                prediction_label,
                float(ckd_probability),
                float(not_ckd_probability),
                model_name,
                input_json
            )
        )

        connection.commit()

        return {
            "id": cursor.lastrowid,
            "created_at": created_at
        }


def get_history_list(limit=100):
    """Lay danh sach lich su, khong tra ve 24 truong du lieu."""

    safe_limit = max(1, min(int(limit), 200))

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                created_at,
                prediction,
                prediction_label,
                ckd_probability,
                not_ckd_probability,
                model_name
            FROM prediction_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,)
        ).fetchall()

    return [dict(row) for row in rows]


def get_history_detail(record_id):
    """Lay chi tiet mot lan phan tich."""

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM prediction_history
            WHERE id = ?
            """,
            (record_id,)
        ).fetchone()

    if row is None:
        return None

    result = dict(row)

    try:
        result["input_data"] = json.loads(
            result["input_data"]
        )
    except (TypeError, json.JSONDecodeError):
        result["input_data"] = {}

    return result


def delete_history(record_id):
    """Xoa mot ban ghi lich su."""

    with get_connection() as connection:
        cursor = connection.execute(
            """
            DELETE FROM prediction_history
            WHERE id = ?
            """,
            (record_id,)
        )

        connection.commit()

        return cursor.rowcount > 0
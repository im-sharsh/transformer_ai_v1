"""
Query engine - loads the clean canonical DataFrame into a real SQLite
database so questions can be answered by EXECUTING a query against actual
data, not by having an LLM guess or memorize a number. This is the
foundation both the NL-to-SQL task and fact-grounded summarization build on.
"""
import sqlite3
import pandas as pd

TABLE_COLUMNS = ["entity_id", "timestamp", "amount", "currency",
                  "category", "channel", "counterparty", "status"]


def build_sqlite_db(df: pd.DataFrame, warn_callback=None) -> sqlite3.Connection:
    """Build an in-memory SQLite database from the clean canonical DataFrame.

    Defensive dedup: clean_df should never have two columns with the same
    canonical name, but if it ever does (a bug upstream we haven't traced,
    or an unusual input we haven't seen), silently keeping the first
    occurrence and warning is far better than a hard crash here - this is
    the last line of defense before the data leaves Python's control and
    hits SQLite directly.
    """
    conn = sqlite3.connect(":memory:")
    export_df = df[TABLE_COLUMNS].copy()

    duplicated_mask = export_df.columns.duplicated()
    if duplicated_mask.any():
        dup_names = sorted(set(export_df.columns[duplicated_mask]))
        message = (
            f"Duplicate column name(s) found in cleaned data just before building "
            f"the query engine: {dup_names}. Keeping only the first occurrence of "
            f"each - this points to a schema-mapping bug upstream, not bad input "
            f"data, and should be reported."
        )
        if warn_callback:
            warn_callback(message)
        else:
            print(f"WARNING: {message}")
        export_df = export_df.loc[:, ~export_df.columns.duplicated()]

    export_df["timestamp"] = export_df["timestamp"].astype(str)
    export_df.to_sql("transactions", conn, index=False, if_exists="replace")
    conn.execute("CREATE INDEX idx_entity_time ON transactions(entity_id, timestamp)")
    conn.commit()
    return conn


def run_query(conn: sqlite3.Connection, sql: str, params: tuple = ()):
    """Execute SQL and return (success, rows_or_error). Never raises -
    a bad query is a normal, expected outcome to check for, not a crash."""
    try:
        cur = conn.execute(sql, params)
        rows = cur.fetchall()
        columns = [d[0] for d in cur.description] if cur.description else []
        return True, {"columns": columns, "rows": rows}
    except Exception as e:
        return False, str(e)

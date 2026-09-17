"""
Stage B - Task-Aware Shaping (LLM-focused adapter)

Takes the clean canonical DataFrame from Stage A and shapes it into
structured, instruction-style JSONL records for one of:
  fraud_flagging | categorization | spend_summarization | general_behavioral

Serialization is structured-but-textual (compact key:value strings), not
natural-language prose - this preserves numeric precision and avoids
burning tokens on boilerplate sentences.
"""
import pandas as pd
import numpy as np

MAX_CONTEXT_TXNS = 10  # simple stand-in for a token-budget-based window


def _serialize_txn(row, exclude_fields=None):
    exclude_fields = exclude_fields or set()
    fields = {
        "date": row["timestamp"].strftime("%Y-%m-%d %H:%M"),
        "amount": f"{row['amount']:.2f}",
        "currency": row.get("currency", "USD"),
        "category": row["category"],
        "channel": row["channel"],
        "counterparty": row["counterparty"],
        "status": row["status"],
    }
    # any opt-in extra context columns (quarantined-but-carried-through) ride
    # along automatically, labeled with their original column name
    for key in row.index:
        if key.startswith("extra__") and pd.notna(row[key]):
            fields[key.replace("extra__", "")] = row[key]
    for f in exclude_fields:
        fields.pop(f, None)
    inner = ", ".join(f"{k}: {v}" for k, v in fields.items())
    return "{" + inner + "}"


def build_fraud_examples(df: pd.DataFrame, max_context=MAX_CONTEXT_TXNS):
    examples = []
    for entity_id, grp in df.groupby("entity_id"):
        grp = grp.sort_values("timestamp")
        for i in range(len(grp)):
            window = grp.iloc[max(0, i - max_context + 1): i + 1]
            label = window.iloc[-1].get("label")
            if pd.isna(label) or label is None:
                continue  # fraud task needs labels; skip unlabeled rows rather than fabricate
            context = [_serialize_txn(r) for _, r in window.iterrows()]
            examples.append({
                "task": "fraud_flagging",
                "instruction": "Given this customer's recent transaction history, determine "
                               "whether the most recent transaction is fraudulent. "
                               "Respond with FRAUD or LEGITIMATE and a brief reason.",
                "context": context,
                "output": "FRAUD" if str(label) in ("1", "True", "true", "FRAUD") else "LEGITIMATE",
                "split": window.iloc[-1].get("split", "unassigned"),
            })
    return examples


def build_categorization_examples(df: pd.DataFrame):
    examples = []
    for _, row in df.iterrows():
        serialized = _serialize_txn(row, exclude_fields={"category"})  # avoid leaking the target
        examples.append({
            "task": "categorization",
            "instruction": "Categorize the following transaction.",
            "context": [serialized],
            "output": row["category"],
            "split": row.get("split", "unassigned"),
        })
    return examples


def build_general_examples(df: pd.DataFrame, max_context=MAX_CONTEXT_TXNS):
    examples = []
    for entity_id, grp in df.groupby("entity_id"):
        grp = grp.sort_values("timestamp")
        if len(grp) < 2:
            continue
        # Multiple sliding windows across the entity's FULL history, not just
        # the single most-recent window. Confirmed in testing: taking only
        # each entity's tail(max_context) transactions means every entity's
        # single window clusters near the END of the whole dataset's time
        # range (since it's each entity's most recent activity) - which,
        # under time-based splitting, landed 100% of examples in the 'test'
        # bucket and left train/val completely empty. Sliding windows spread
        # naturally across the full timeline instead.
        step = max(1, max_context // 2)  # 50% overlap between consecutive windows
        window_ends = list(range(2, len(grp), step)) or []
        window_ends.append(len(grp))  # always include the final window too
        seen_ends = set()
        for end in window_ends:
            if end in seen_ends or end < 2:
                continue
            seen_ends.add(end)
            window = grp.iloc[max(0, end - max_context):end]
            sequence = [_serialize_txn(r) for _, r in window.iterrows()]
            examples.append({
                "task": "general_behavioral",
                "sequence": sequence,
                "split": window.iloc[-1].get("split", "unassigned"),
            })
    return examples


TASK_BUILDERS = {
    "fraud_flagging": build_fraud_examples,
    "categorization": build_categorization_examples,
    "general_behavioral": build_general_examples,
    # NOTE: "spend_summarization" and "query_sql" are NOT here - they're
    # routed through the query engine in orchestrator.py instead, since they
    # need a real SQLite connection to compute/validate facts and queries.
}


def run_stage_b(df: pd.DataFrame, task_type: str):
    builder = TASK_BUILDERS[task_type]
    examples = builder(df)
    stats = {
        "task_type": task_type,
        "example_count": len(examples),
        "split_distribution": pd.Series([e.get("split", "unassigned") for e in examples]).value_counts().to_dict(),
    }
    if task_type == "fraud_flagging":
        skipped = df["label"].isna().sum()
        stats["skipped_unlabeled_rows"] = int(skipped)
        if len(examples) > 0:
            fraud_ct = sum(1 for e in examples if e["output"] == "FRAUD")
            stats["label_distribution"] = {"FRAUD": fraud_ct, "LEGITIMATE": len(examples) - fraud_ct}
    return examples, stats

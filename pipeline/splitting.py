"""
Time-based splitting.

Never split transaction data randomly - a model evaluated on randomly
shuffled data can effectively "see the future," inflating scores in a way
that doesn't hold up in production. Instead, pick cutoff timestamps: earlier
data trains, later data tests - exactly like real deployment.
"""
import pandas as pd


def compute_time_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    """
    Assigns each row a 'split' label (train/val/test) based on global
    chronological cutoffs - NOT per-entity, so all entities share the same
    train/val/test time boundaries (this is what makes the split meaningful:
    "everything after this date is unseen," consistently, across the dataset).

    Returns (df_with_split_column, split_stats_dict).
    """
    assert 0 < train_frac < 1 and 0 <= val_frac <= 1 - train_frac, \
        "train_frac + val_frac must be <= 1"

    df = df.copy()
    ordered = df.sort_values("timestamp")
    n = len(ordered)

    train_end_idx = int(n * train_frac)
    val_end_idx = int(n * (train_frac + val_frac))

    train_cutoff = ordered.iloc[min(train_end_idx, n - 1)]["timestamp"]
    val_cutoff = ordered.iloc[min(val_end_idx, n - 1)]["timestamp"]

    def _label(ts):
        if ts <= train_cutoff:
            return "train"
        elif ts <= val_cutoff:
            return "val"
        return "test"

    df["split"] = df["timestamp"].apply(_label)

    counts = df["split"].value_counts().to_dict()
    stats = {
        "train_frac_requested": train_frac,
        "val_frac_requested": val_frac,
        "train_cutoff": str(train_cutoff),
        "val_cutoff": str(val_cutoff),
        "row_counts": {k: int(counts.get(k, 0)) for k in ["train", "val", "test"]},
    }
    return df, stats

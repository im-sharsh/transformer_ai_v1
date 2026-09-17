"""
Stage A - Canonical Cleaning (transformer-agnostic)

Applies the resolved schema mapping, validates data quality, hashes PII,
normalizes conventions, and computes causal per-entity features.
Output is a clean canonical DataFrame - identical regardless of what
downstream transformer/task will consume it.
"""
import hashlib
import re
import pandas as pd
import numpy as np

from .schema import CANONICAL_FIELDS, REQUIRED_FIELDS
from .schema_resolution import safe_to_datetime

PII_FIELDS = {"entity_id", "counterparty"}
_SALT = "txn-pipeline-fixed-salt-v1"  # fixed so entity linkage is preserved across a run

_CURRENCY_SYMBOLS = r"[$€£₹¥]"
_CURRENCY_CODES = r"\b(?:USD|EUR|GBP|INR|JPY|CAD|AUD|CHF)\b"


def _normalize_amount_string(raw):
    """
    Turn a messy amount string into something pd.to_numeric can parse
    correctly - or return NaN rather than risk a WRONG number.

    Handles, in order:
      - currency symbols ($€£₹¥) and 3-letter currency codes (USD, EUR, ...)
        as prefix or suffix
      - parentheses-negative accounting convention: (123.45) -> -123.45
      - trailing-minus convention (common in mainframe/core-banking exports):
        123.45- -> -123.45
      - European vs. US thousands/decimal separator convention - this is the
        critical one: naively stripping all commas turns the European
        "1.234,56" (meaning 1234.56) into "1.23456", a WRONG number that
        would parse "successfully" and pollute training data silently. This
        is distinguished by checking whether a comma or a period appears
        LAST in the string (whichever is last is the decimal separator).
    """
    if pd.isna(raw):
        return raw
    s = str(raw).strip()
    if s == "":
        return None

    s = re.sub(_CURRENCY_SYMBOLS, "", s)
    s = re.sub(_CURRENCY_CODES, "", s, flags=re.IGNORECASE).strip()

    paren_match = re.match(r"^\((.*)\)$", s)
    if paren_match:
        s = "-" + paren_match.group(1)

    if s.endswith("-"):
        s = "-" + s[:-1]

    s = s.strip()

    has_comma, has_period = "," in s, "." in s
    if has_comma and has_period:
        # whichever separator appears LAST is the decimal point
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")  # European: 1.234,56 -> 1234.56
        else:
            s = s.replace(",", "")                      # US: 1,234.56 -> 1234.56
    elif has_comma and not has_period:
        # ambiguous alone (1,000 could be thousands-sep or a European decimal),
        # but a comma followed by exactly 2 digits at the end is far more
        # likely a decimal separator (e.g. "1234,56") than a thousands group
        if re.match(r"^-?\d+,\d{2}$", s):
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")

    return s


def _hash_id(value) -> str:
    if pd.isna(value):
        return value
    return hashlib.sha256(f"{_SALT}:{value}".encode()).hexdigest()[:16]


def apply_schema_mapping(df: pd.DataFrame, column_map: dict, direction_col: str,
                          amount_assumptions: list, extra_context_cols: list = None,
                          warn_callback=None) -> pd.DataFrame:
    """Rename mapped columns to canonical names, fill optional defaults.

    extra_context_cols: optional list of RAW (quarantined) column names the
    user explicitly opted to carry through as extra context, even though
    they don't fit a canonical field. Carried through under an 'extra__'
    prefix, UNHASHED - these bypass the PII-hashing applied to canonical
    entity_id/counterparty, so this is opt-in and the caller is responsible
    for warning the user before selecting anything that might be sensitive.
    """
    rename_map = {src: canon for src, canon in column_map.items()}

    # Defensive check #1: column_map itself should never have two different
    # SOURCE columns mapping to the same canonical field. This catches
    # mapping-logic bugs (e.g. an override that didn't properly supersede a
    # prior mapping) with a clear, attributable error message.
    seen_targets = {}
    for src, canon in rename_map.items():
        if canon in seen_targets:
            raise ValueError(
                f"Schema mapping conflict: both '{seen_targets[canon]}' and '{src}' are "
                f"mapped to canonical field '{canon}'. A canonical field can only come from "
                f"one source column - this is a bug in the mapping step, not the input data."
            )
        seen_targets[canon] = src

    clean = df.rename(columns=rename_map)[list(rename_map.values())].copy()

    # Defensive check #2: even with (1) passing, the INPUT dataframe could in
    # rare cases already carry genuinely duplicate-labeled columns for a
    # mapped name (seen in testing: this bypasses the column_map-level check
    # entirely, since dict keys can't duplicate but a pandas column Index
    # can). Rather than let this surface as a cryptic SQLite error several
    # stages later, catch and fix it here: keep the first occurrence, warn.
    if clean.columns.duplicated().any():
        dup_names = sorted(set(clean.columns[clean.columns.duplicated()]))
        message = (
            f"Duplicate column name(s) after schema mapping: {dup_names}. Keeping only "
            f"the first occurrence of each. This means the input data (or an upstream "
            f"mapping step) produced more than one column resolving to the same "
            f"canonical field - please report this if you see it, including the "
            f"original column headers."
        )
        if warn_callback:
            warn_callback(message)
        else:
            print(f"WARNING: {message}")
        clean = clean.loc[:, ~clean.columns.duplicated()]

    # apply cents -> base unit conversion if flagged
    if "amount" in clean.columns:
        # NOTE: check is_numeric_dtype rather than `dtype == object` - pandas
        # 2.x/3.x may represent text columns as a dedicated 'str' dtype that
        # isn't numpy object, and a naive equality check would silently skip
        # this cleanup step entirely.
        if not pd.api.types.is_numeric_dtype(clean["amount"]):
            clean["amount"] = clean["amount"].apply(_normalize_amount_string)
        clean["amount"] = pd.to_numeric(clean["amount"], errors="coerce")
        for a in amount_assumptions:
            if a["field"] == "amount" and "cents" in a["assumption"]:
                clean["amount"] = clean["amount"] / 100.0

        # apply sign from direction column if present
        if direction_col is not None and direction_col in df.columns:
            direction = df[direction_col].astype(str).str.upper().str.strip()
            debit_like = direction.isin(["DR", "DEBIT", "D", "-"])
            clean["amount"] = np.where(debit_like, -clean["amount"].abs(), clean["amount"].abs())

    if "timestamp" in clean.columns:
        dayfirst = True
        for a in amount_assumptions:
            if a["field"] == "timestamp" and "_dayfirst" in a:
                dayfirst = a["_dayfirst"]
        clean["timestamp"] = safe_to_datetime(clean["timestamp"], errors="coerce",
                                               dayfirst=dayfirst, format="mixed")

    for field, spec in CANONICAL_FIELDS.items():
        if field not in clean.columns:
            clean[field] = spec.get("default")

    for col in (extra_context_cols or []):
        if col in df.columns:
            clean[f"extra__{col}"] = df[col].values

    return clean


def validate_and_gate(df: pd.DataFrame):
    """Hard-gate rows missing required fields; report duplicates.
    Also reports per-field null counts BEFORE dropping, so a mass drop is
    diagnosable (which field failed) instead of a mystery total."""
    stats = {"input_rows": len(df)}

    required_present = [f for f in REQUIRED_FIELDS if f in df.columns]
    stats["null_counts_by_required_field"] = {
        f: int(df[f].isna().sum()) for f in required_present
    }

    before = len(df)
    df = df.dropna(subset=required_present)
    stats["dropped_missing_required"] = before - len(df)

    dup_mask = df.duplicated(subset=["entity_id", "timestamp", "amount", "counterparty"], keep="first")
    stats["duplicate_rows_removed"] = int(dup_mask.sum())
    df = df[~dup_mask]

    stats["output_rows"] = len(df)
    return df, stats


def hash_pii(df: pd.DataFrame):
    """Hash identifier fields. Returns df with hashed ids + list of fields hashed."""
    df = df.copy()
    hashed_fields = []
    for field in PII_FIELDS:
        if field in df.columns:
            df[field] = df[field].apply(_hash_id)
            hashed_fields.append(field)
    return df, hashed_fields


def add_causal_features(df: pd.DataFrame):
    """Per-entity, chronologically causal derived features. Never looks forward."""
    df = df.sort_values(["entity_id", "timestamp"]).copy()

    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour_of_day"] = df["timestamp"].dt.hour

    grp = df.groupby("entity_id")["timestamp"]
    df["time_since_last_txn_hours"] = (
        grp.diff().dt.total_seconds() / 3600.0
    )

    df["rolling_avg_amount_3"] = (
        df.groupby("entity_id")["amount"]
          .apply(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
          .reset_index(level=0, drop=True)
    )

    return df


def run_stage_a(raw_df, column_map, direction_col, amount_assumptions, extra_context_cols=None,
                 warn_callback=None):
    """Full Stage A pipeline. Returns (clean_df, stats_dict)."""
    mapped = apply_schema_mapping(raw_df, column_map, direction_col, amount_assumptions,
                                   extra_context_cols, warn_callback=warn_callback)
    gated, gate_stats = validate_and_gate(mapped)
    hashed, hashed_fields = hash_pii(gated)
    featured = add_causal_features(hashed)

    stats = {
        **gate_stats,
        "pii_fields_hashed": hashed_fields,
    }
    return featured, stats

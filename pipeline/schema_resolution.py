"""
Stage 0 - Schema Resolution

Takes an arbitrary raw transaction DataFrame and maps its columns onto the
canonical schema, using:
  1. Name-based matching (exact + fuzzy against alias dictionary)
  2. Content-based inference as a fallback for unhelpful column names
  3. Confidence scoring, so ambiguous mappings are flagged, not guessed silently

It also detects common unit/convention ambiguities (cents vs dollars,
sign convention, date format) and records the assumption applied.
"""
import re
import difflib
import pandas as pd
import numpy as np

from .schema import ALIAS_DICTIONARY, CANONICAL_FIELDS


def safe_to_datetime(series, **kwargs):
    """
    pd.to_datetime crashes outright (unhandled ValueError) when a column mixes
    timezone-aware values (e.g. '2026-01-05T14:30:00Z' or '...+05:30') with
    timezone-naive ones - a common real-world occurrence when data is merged
    from multiple export sources. Rather than let this take down the whole
    pipeline, retry with utc=True and normalize to naive timestamps - the rest
    of this pipeline (splitting, causal features, SQL queries) assumes naive
    timestamps throughout, so we standardize here rather than carry timezone
    awareness through inconsistently.
    """
    try:
        return pd.to_datetime(series, **kwargs)
    except ValueError as e:
        if "Mixed timezones" not in str(e) and "tz-aware" not in str(e):
            raise
        kwargs_utc = {k: v for k, v in kwargs.items() if k != "dayfirst"}
        parsed = pd.to_datetime(series, utc=True, **{k: v for k, v in kwargs.items() if k in ("errors", "format")})
        return parsed.dt.tz_localize(None)


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _best_name_match(col_name: str):
    """Return (canonical_field, confidence, method) for a single column name."""
    norm = _normalize(col_name)
    best_field, best_score, best_method = None, 0.0, None

    for field, aliases in ALIAS_DICTIONARY.items():
        for alias in aliases:
            if norm == alias:
                return field, 0.95, "exact_alias"
            ratio = difflib.SequenceMatcher(None, norm, alias).ratio()
            if ratio > best_score:
                best_field, best_score, best_method = field, ratio, "fuzzy_alias"

    if best_score >= 0.80:
        return best_field, round(0.75 * best_score, 2), best_method
    return None, 0.0, None


def _infer_from_content(series: pd.Series):
    """Best-effort guess of field kind from values alone. Used only when
    name-based matching fails or is ambiguous. Returns (kind, confidence)."""
    sample = series.dropna().head(200)
    if len(sample) == 0:
        return None, 0.0

    # datetime?
    parsed = safe_to_datetime(sample, errors="coerce", format="mixed")
    if parsed.notna().mean() > 0.9:
        return "timestamp", 0.55

    # numeric?
    numeric = pd.to_numeric(sample, errors="coerce")
    if numeric.notna().mean() > 0.9:
        return "amount", 0.5

    # id-like (high cardinality strings)?
    if not pd.api.types.is_numeric_dtype(sample) and sample.nunique() / len(sample) > 0.8:
        return "entity_id", 0.4

    # low-cardinality categorical - can't safely tell which canonical field
    if sample.nunique() <= max(10, len(sample) * 0.2):
        return "unmapped_categorical", 0.3

    return None, 0.0


def _content_plausible_for_field(series: pd.Series, field: str) -> bool:
    """
    Guard against fuzzy NAME matches that are semantically wrong despite
    looking similar as strings (e.g. 'transaction_id' fuzzy-matching the
    'timestamp' alias 'transactiondate'). Only applied to fuzzy matches -
    exact alias matches are trusted as-is.
    """
    sample = series.dropna().astype(str).head(200)
    if len(sample) == 0:
        return True  # nothing to contradict the guess

    if field == "timestamp":
        parsed = safe_to_datetime(sample, errors="coerce", format="mixed")
        return parsed.notna().mean() > 0.7

    if field == "amount":
        cleaned = sample.str.replace(r"[,$€£\s]", "", regex=True)
        numeric = pd.to_numeric(cleaned, errors="coerce")
        return numeric.notna().mean() > 0.7

    return True  # no strong content signature to check for id/categorical fields


def resolve_schema(df: pd.DataFrame):
    """
    Returns:
        mapping_report: list of dicts describing how each source column resolved
        column_map: {source_col: canonical_field} for columns mapped with
                    sufficient confidence
        quarantined_columns: list of source columns left unmapped
        direction_col: source column name to use for sign resolution, or None

    IMPORTANT: resolution happens in two confidence-ordered passes, not in
    raw column order. A weak content-inference guess (e.g. "this column of
    unique strings is probably entity_id") must never be allowed to claim a
    canonical field ahead of a later column that's an exact name match for
    that same field - otherwise the outcome depends on column order, which
    is exactly the kind of silent, hard-to-notice bug this pipeline is
    supposed to avoid. Pass 1 resolves ALL name-based matches globally,
    strongest first; only leftover columns/fields go to Pass 2 (content-only
    inference).
    """
    direction_col = None
    used_canonical = set()
    resolved = {}  # col -> (field, confidence, method, status, note)

    # ---- Pass 1: name-based matches, strongest confidence claims first ----
    name_candidates = []  # (confidence, col, field, method)
    for col in df.columns:
        field, confidence, method = _best_name_match(col)
        if field == "_direction":
            direction_col = col
            resolved[col] = ("_direction (sign helper)", confidence, method, "used_for_sign_resolution", None)
            continue
        if field is not None and confidence >= 0.55:
            name_candidates.append((confidence, col, field, method))

    # highest-confidence name match wins the field first, regardless of
    # which column happened to appear earlier in the raw file
    for confidence, col, field, method in sorted(name_candidates, key=lambda x: -x[0]):
        if field in used_canonical:
            # another column already claimed this field at >= this confidence -
            # surface it explicitly rather than silently quarantining the loser,
            # which is deeply confusing when the loser's name looks like an
            # equally (or more) obvious match (e.g. a literal 'channel' column
            # losing to 'payment_method' just because of alias-list overlap).
            # Resolved directly here (not sent to Pass 2) - a column that just
            # lost a strong name-match tie shouldn't then get reinterpreted by
            # a much weaker content guess into some unrelated field.
            note = (
                f"Also matched canonical field '{field}' (confidence {confidence:.2f}), but "
                f"that field was already claimed by another column. If '{col}' is actually "
                f"the better match, use the field-override tool to fix this."
            )
            resolved[col] = (None, confidence, method, "quarantined_collision", note)
            continue
        if col in resolved:
            continue
        if method == "fuzzy_alias" and not _content_plausible_for_field(df[col], field):
            continue  # name looked similar, content doesn't back it up - leave for Pass 2
        status = "auto_mapped" if confidence >= 0.85 else "auto_mapped_flagged"
        resolved[col] = (field, confidence, method, status, None)
        used_canonical.add(field)

    # ---- Pass 2: content-based inference for whatever's left ----
    content_candidates = []  # (confidence, col, kind)
    for col in df.columns:
        if col in resolved:
            continue
        kind, c_confidence = _infer_from_content(df[col])
        if kind in ("timestamp", "amount", "entity_id") and c_confidence >= 0.4:
            content_candidates.append((c_confidence, col, kind))
        else:
            resolved[col] = (None, c_confidence, "content_inference", "quarantined", None)

    for c_confidence, col, kind in sorted(content_candidates, key=lambda x: -x[0]):
        if col in resolved:
            continue
        if kind not in used_canonical:
            resolved[col] = (kind, c_confidence, "content_inference", "auto_mapped_flagged", None)
            used_canonical.add(kind)
        else:
            resolved[col] = (None, c_confidence, "content_inference", "quarantined", None)

    # ---- assemble outputs in original column order ----
    mapping_report = []
    column_map = {}
    quarantined_columns = []
    for col in df.columns:
        field, confidence, method, status, note = resolved[col]
        entry = {
            "source_column": col, "canonical_field": field,
            "confidence": confidence, "method": method, "status": status,
        }
        if note:
            entry["note"] = note
        mapping_report.append(entry)
        if status in ("quarantined", "quarantined_collision"):
            quarantined_columns.append(col)
        elif status != "used_for_sign_resolution":
            column_map[col] = field

    return mapping_report, column_map, quarantined_columns, direction_col


def detect_conventions(df: pd.DataFrame, column_map: dict, direction_col: str):
    """
    Detect amount scale (cents vs dollars) and date format ambiguity.
    Returns a list of assumption dicts to be logged in the manifest.
    """
    assumptions = []
    amount_col = next((src for src, canon in column_map.items() if canon == "amount"), None)
    ts_col = next((src for src, canon in column_map.items() if canon == "timestamp"), None)

    if amount_col:
        norm_name = _normalize(amount_col)
        vals = pd.to_numeric(df[amount_col], errors="coerce").dropna()
        looks_like_cents = "cent" in norm_name or (
            (vals % 1 == 0).mean() > 0.98 and vals.median() > 1000
        )
        if looks_like_cents:
            assumptions.append({
                "field": "amount",
                "assumption": "values appear to be in cents; converted to base units (divided by 100)",
                "confidence": "medium" if "cent" not in norm_name else "high",
            })
        if direction_col is not None:
            assumptions.append({
                "field": "amount",
                "assumption": f"sign derived from separate direction column '{direction_col}' "
                               f"(debit-like values -> negative, credit-like -> positive)",
                "confidence": "high",
            })
        elif (vals < 0).sum() == 0:
            assumptions.append({
                "field": "amount",
                "assumption": "no negative values and no direction column found; "
                               "all amounts treated as unsigned spend amounts",
                "confidence": "low - flagged for review",
            })

    if ts_col:
        sample = df[ts_col].dropna().astype(str).head(300)
        dayfirst_resolved, date_note = _resolve_date_format(sample)
        assumptions.append({
            "field": "timestamp",
            "assumption": date_note,
            "confidence": "high" if "ISO" in date_note else (
                "low - flagged for review" if "ambiguous" in date_note else "medium"
            ),
            "_dayfirst": dayfirst_resolved,  # consumed by cleaning stage; not a display field
        })

    return assumptions


def _resolve_date_format(sample: pd.Series):
    """
    Decide whether to parse with dayfirst=True or False, without corrupting
    unambiguous ISO (YYYY-MM-DD) strings - pandas' dayfirst flag can otherwise
    incorrectly swap day/month even on ISO-formatted input.
    Returns (dayfirst_bool, human_readable_note).
    """
    if sample.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.9:
        return False, "ISO date format (YYYY-MM-DD) detected; parsed unambiguously"

    dayfirst_parse = safe_to_datetime(sample, errors="coerce", dayfirst=True, format="mixed")
    monthfirst_parse = safe_to_datetime(sample, errors="coerce", dayfirst=False, format="mixed")
    day_gt_12_dayfirst = any(d.day > 12 for d in dayfirst_parse.dropna())
    day_gt_12_monthfirst = any(d.day > 12 for d in monthfirst_parse.dropna())

    if day_gt_12_monthfirst and not day_gt_12_dayfirst:
        return True, "day values >12 found under month-first parsing; format resolved as day-first"
    if day_gt_12_dayfirst and not day_gt_12_monthfirst:
        return False, "day values >12 found under day-first parsing; format resolved as month-first"
    return True, "date format ambiguous (DD/MM vs MM/DD indistinguishable from data); " \
                 "defaulted to day-first parsing"

from .orchestrator import run_pipeline, AUTO_INDEX_ENTITY_ID
from .schema import CANONICAL_FIELDS
from .schema_resolution import resolve_schema

REQUIRED_CANONICAL = {"entity_id", "timestamp", "amount"}
IMPORTANT_OPTIONAL_CANONICAL = {"category", "label"}  # defaulting silently here can quietly empty out a task's output


def check_missing_fields(raw_df):
    """Quick pre-check: which fields couldn't be auto-mapped.
    Returns (missing_required, missing_important_optional, column_map, candidate_columns).

    missing_required blocks the pipeline if left unresolved.
    missing_important_optional does NOT block the pipeline (it has a
    default), but silently defaulting 'category' or 'label' can quietly
    empty out the categorization/fraud_flagging task output with no
    warning - so the user gets a chance to map these explicitly too."""
    _, column_map, quarantined_columns, _ = resolve_schema(raw_df)
    mapped_fields = set(column_map.values())
    missing_required = REQUIRED_CANONICAL - mapped_fields
    missing_important_optional = IMPORTANT_OPTIONAL_CANONICAL - mapped_fields
    candidates = [c for c in raw_df.columns if c not in column_map]
    return missing_required, missing_important_optional, column_map, candidates


def check_missing_required(raw_df):
    """Back-compat wrapper - see check_missing_fields for the fuller check."""
    missing_required, _, column_map, candidates = check_missing_fields(raw_df)
    return missing_required, column_map, candidates


__all__ = ["run_pipeline", "CANONICAL_FIELDS", "check_missing_fields",
           "check_missing_required", "AUTO_INDEX_ENTITY_ID"]



"""
NL-to-SQL task shaping.

Rather than teaching a model to compute answers directly (memorization-prone
and arithmetic-unreliable), this teaches translation: natural-language
question -> SQL query. The query then gets executed for real against the
SQLite engine, so the "ground truth" is always a real, current answer, not
a frozen guess baked into the model's weights.

Coverage is combinatorial, not hand-enumerated: a small set of template
FAMILIES (aggregation, extremes, comparison, retrieval) is instantiated
against whatever categories/time-windows/entities actually exist in THIS
dataset. Every generated SQL query is executed and validated before being
kept - a query that errors or returns nothing useful is dropped, not
included as a bad example.

Scope: "my spending" queries, scoped to a single entity - not
cross-entity/analyst-style aggregate queries.
"""
import random
import calendar
from datetime import datetime
from .query_engine import run_query


def _month_windows(dates):
    """Given a sorted list of datetimes, return distinct (year, month) buckets
    present, as (label, start_iso, end_iso) tuples."""
    seen = {}
    for d in dates:
        key = (d.year, d.month)
        if key not in seen:
            last_day = calendar.monthrange(d.year, d.month)[1]
            start = datetime(d.year, d.month, 1)
            end = datetime(d.year, d.month, last_day, 23, 59, 59)
            label = start.strftime("%B %Y")
            seen[key] = (label, start.isoformat(), end.isoformat())
    return list(seen.values())


TEMPLATES = []  # populated below by decorator-style registration


def _register(kind):
    def deco(fn):
        TEMPLATES.append((kind, fn))
        return fn
    return deco


# ---- Aggregation ----

@_register("aggregation")
def _tpl_total_spend_by_category(entity_id, categories, windows, rng):
    if not categories or not windows:
        return None
    cat = rng.choice(categories)
    label, start, end = rng.choice(windows)
    variants = [
        f"How much did I spend on {cat} in {label}?",
        f"What did I spend on {cat} during {label}?",
        f"Total {cat} spending for {label}?",
    ]
    sql = ("SELECT SUM(ABS(amount)) FROM transactions "
           "WHERE entity_id=? AND category=? AND amount<0 AND timestamp BETWEEN ? AND ?")
    return variants, sql, (entity_id, cat, start, end), "scalar"


@_register("aggregation")
def _tpl_total_spend_overall(entity_id, categories, windows, rng):
    if not windows:
        return None
    label, start, end = rng.choice(windows)
    variants = [
        f"How much did I spend in {label}?",
        f"What was my total spending in {label}?",
        f"How much money went out in {label}?",
    ]
    sql = ("SELECT SUM(ABS(amount)) FROM transactions "
           "WHERE entity_id=? AND amount<0 AND timestamp BETWEEN ? AND ?")
    return variants, sql, (entity_id, start, end), "scalar"


@_register("aggregation")
def _tpl_count_transactions(entity_id, categories, windows, rng):
    if not windows:
        return None
    label, start, end = rng.choice(windows)
    variants = [
        f"How many transactions did I make in {label}?",
        f"How many purchases did I have in {label}?",
        f"Number of transactions in {label}?",
    ]
    sql = ("SELECT COUNT(*) FROM transactions "
           "WHERE entity_id=? AND amount<0 AND timestamp BETWEEN ? AND ?")
    return variants, sql, (entity_id, start, end), "count"


@_register("aggregation")
def _tpl_average_transaction(entity_id, categories, windows, rng):
    if not windows:
        return None
    label, start, end = rng.choice(windows)
    variants = [
        f"What was my average transaction amount in {label}?",
        f"On average, how much was each purchase in {label}?",
        f"Average spend per transaction, {label}?",
    ]
    sql = ("SELECT AVG(ABS(amount)) FROM transactions "
           "WHERE entity_id=? AND amount<0 AND timestamp BETWEEN ? AND ?")
    return variants, sql, (entity_id, start, end), "scalar"


@_register("aggregation")
def _tpl_top_category(entity_id, categories, windows, rng):
    if not windows:
        return None
    label, start, end = rng.choice(windows)
    variants = [
        f"What did I spend the most on in {label}?",
        f"My top spending category in {label}?",
        f"Which category did I spend the most in during {label}?",
    ]
    sql = ("SELECT category, SUM(ABS(amount)) as total FROM transactions "
           "WHERE entity_id=? AND amount<0 AND timestamp BETWEEN ? AND ? "
           "GROUP BY category ORDER BY total DESC LIMIT 1")
    return variants, sql, (entity_id, start, end), "row"


# ---- Extremes ----

@_register("extremes")
def _tpl_largest_transaction(entity_id, categories, windows, rng):
    if not windows:
        return None
    label, start, end = rng.choice(windows)
    variants = [
        f"What was my largest transaction in {label}?",
        f"Biggest purchase I made in {label}?",
        f"What's the highest amount I spent on a single transaction in {label}?",
    ]
    sql = ("SELECT timestamp, amount, category, counterparty FROM transactions "
           "WHERE entity_id=? AND amount<0 AND timestamp BETWEEN ? AND ? "
           "ORDER BY amount ASC LIMIT 1")
    return variants, sql, (entity_id, start, end), "row"


@_register("extremes")
def _tpl_smallest_transaction(entity_id, categories, windows, rng):
    if not windows:
        return None
    label, start, end = rng.choice(windows)
    variants = [
        f"What was my smallest transaction in {label}?",
        f"Smallest purchase I made in {label}?",
        f"What's the lowest amount I spent on a single transaction in {label}?",
    ]
    sql = ("SELECT timestamp, amount, category, counterparty FROM transactions "
           "WHERE entity_id=? AND amount<0 AND timestamp BETWEEN ? AND ? "
           "ORDER BY amount DESC LIMIT 1")
    return variants, sql, (entity_id, start, end), "row"


# ---- Comparison ----

@_register("comparison")
def _tpl_compare_categories(entity_id, categories, windows, rng):
    if len(categories) < 2 or not windows:
        return None
    cat_a, cat_b = rng.sample(categories, 2)
    label, start, end = rng.choice(windows)
    variants = [
        f"Did I spend more on {cat_a} or {cat_b} in {label}?",
        f"Which did I spend more on in {label}: {cat_a} or {cat_b}?",
        f"Compare my {cat_a} and {cat_b} spending in {label}.",
    ]
    sql = ("SELECT category, SUM(ABS(amount)) as total FROM transactions "
           "WHERE entity_id=? AND amount<0 AND category IN (?,?) AND timestamp BETWEEN ? AND ? "
           "GROUP BY category")
    return variants, sql, (entity_id, cat_a, cat_b, start, end), "compare_rows"


@_register("comparison")
def _tpl_compare_periods(entity_id, categories, windows, rng):
    if len(windows) < 2:
        return None
    (label1, start1, end1), (label2, start2, end2) = rng.sample(windows, 2)
    variants = [
        f"Did I spend more in {label1} or {label2}?",
        f"Which month did I spend more in: {label1} or {label2}?",
        f"Compare my spending between {label1} and {label2}.",
    ]
    sql = ("SELECT "
           "(SELECT SUM(ABS(amount)) FROM transactions WHERE entity_id=? AND amount<0 AND timestamp BETWEEN ? AND ?) as p1, "
           "(SELECT SUM(ABS(amount)) FROM transactions WHERE entity_id=? AND amount<0 AND timestamp BETWEEN ? AND ?) as p2")
    return variants, sql, (entity_id, start1, end1, entity_id, start2, end2), "period_compare"


# ---- Retrieval ----

@_register("retrieval")
def _tpl_list_between_dates(entity_id, categories, windows, rng):
    if not windows:
        return None
    label, start, end = rng.choice(windows)
    variants = [
        f"Show me my transactions in {label}.",
        f"List my transactions from {label}.",
        f"What transactions did I make in {label}?",
    ]
    sql = ("SELECT timestamp, amount, category, counterparty FROM transactions "
           "WHERE entity_id=? AND timestamp BETWEEN ? AND ? ORDER BY timestamp")
    return variants, sql, (entity_id, start, end), "table"


@_register("retrieval")
def _tpl_list_by_category(entity_id, categories, windows, rng):
    if not categories or not windows:
        return None
    cat = rng.choice(categories)
    label, start, end = rng.choice(windows)
    variants = [
        f"Show me my {cat} transactions in {label}.",
        f"List all {cat} purchases from {label}.",
        f"What {cat} transactions did I have in {label}?",
    ]
    sql = ("SELECT timestamp, amount, counterparty FROM transactions "
           "WHERE entity_id=? AND category=? AND timestamp BETWEEN ? AND ? ORDER BY timestamp")
    return variants, sql, (entity_id, cat, start, end), "table"


def _render_literal_sql(sql_template: str, params: tuple) -> str:
    """
    Substitute '?' placeholders with properly-quoted literal values, producing
    SQL that's directly executable on its own - this is what the model
    should actually be trained to produce, since at inference time nothing
    exists to bind '?' placeholders to real values. The parameterized form
    (sql_template + params) is still used internally for safe execution
    during generation/validation - only the TRAINING TARGET changes to the
    literal form.
    """
    parts = sql_template.split("?")
    assert len(parts) - 1 == len(params), "placeholder/param count mismatch"
    rendered = parts[0]
    for param, next_part in zip(params, parts[1:]):
        if isinstance(param, str):
            escaped = param.replace("'", "''")  # basic SQL string escaping
            literal = f"'{escaped}'"
        else:
            literal = str(param)
        rendered += literal + next_part
    return rendered


def _format_reference_answer(kind, result_rows, columns):
    """A deterministic, human-readable rendering of the executed result -
    NOT the fine-tuning target itself, just useful for review in the UI."""
    if not result_rows:
        return "No matching transactions found."
    if kind == "scalar":
        val = result_rows[0][0]
        return "No data." if val is None else f"{val:.2f}"
    if kind == "count":
        val = result_rows[0][0]
        return "No data." if val is None else str(int(val))  # a count is a whole number, never "12.00"
    if kind == "row":
        return str(dict(zip(columns, result_rows[0])))
    if kind == "compare_rows":
        return "; ".join(f"{r[0]}: {r[1]:.2f}" for r in result_rows)
    if kind == "period_compare":
        p1, p2 = result_rows[0]
        return f"period 1: {p1}, period 2: {p2}" if p1 is not None and p2 is not None else "insufficient data"
    if kind == "table":
        return f"{len(result_rows)} matching row(s)"
    return str(result_rows)


def _result_is_usable(kind, result_rows):
    """Skip examples where the executed query didn't actually return a
    meaningful answer - an empty/null result isn't a useful training pair
    for a task about answering real questions with real data."""
    if not result_rows:
        return False
    if kind == "scalar":
        return result_rows[0][0] is not None
    if kind == "count":
        # a genuine zero-spend window can happen (e.g. the only activity
        # that period was a refund, not a purchase), but it's a confusing
        # training example: the window exists because SOMETHING happened
        # that period, yet the "answer" reads as if nothing did. Require a
        # real positive count so "how many transactions" always corresponds
        # to a period with actual counted activity.
        return result_rows[0][0] is not None and result_rows[0][0] > 0
    if kind == "row":
        return True
    if kind == "compare_rows":
        return len(result_rows) == 2 and all(r[1] is not None for r in result_rows)
    if kind == "period_compare":
        p1, p2 = result_rows[0]
        return p1 is not None and p2 is not None
    if kind == "table":
        return len(result_rows) > 0
    return False


def generate_nl_sql_examples(clean_df, conn, split_cutoffs, phrasing_variants=3,
                              max_per_entity=6, seed=0):
    """
    clean_df: the Stage A output (used to discover per-entity categories/windows)
    conn: sqlite3 connection from build_sqlite_db(clean_df)
    split_cutoffs: (train_cutoff, val_cutoff) as datetime - used to label each
                   generated example by which split its time window falls into
    Returns a list of shaped examples, each execution-validated against real data.
    """
    rng = random.Random(seed)
    train_cutoff, val_cutoff = split_cutoffs
    examples = []

    # Every template's SQL hardcodes "amount<0" to mean "this row is spend",
    # matching this pipeline's own convention (signed amounts, spend
    # negative). Some real-world sources never get flipped to that
    # convention by Stage A - e.g. no direction column AND no negative
    # values at all, which schema_resolution.py can only treat as "unsigned
    # spend amounts, low confidence" (see detect_conventions). For such a
    # source, "amount<0" matches ZERO rows: every aggregation/extremes/
    # comparison template silently produced no usable result and got
    # dropped by _result_is_usable, leaving ONLY the two amount-agnostic
    # retrieval templates - confirmed by running this exact function
    # against a real unsigned-amount CSV (diverse_transactions.csv):
    # 1155/1155 generated examples were "retrieval", 0 of every other kind.
    # Detect the convention once per dataset and adapt both the spend
    # filter and the largest/smallest ORDER BY direction (which was
    # written assuming negative-for-spend, so a naive "just drop the
    # filter" fix would silently invert largest/smallest on unsigned data).
    has_negative_amounts = (clean_df["amount"] < 0).any()
    spend_filter = "amount<0" if has_negative_amounts else "1=1"

    # Real-world sources routinely mix currencies per customer (confirmed on
    # a real test CSV: ALL 150 entities had 2+ currencies) and carry
    # non-settled transaction states (Failed/Pending/Refunded, alongside
    # Completed) with no special handling anywhere in these templates.
    # Every SUM/AVG/extremes/comparison query was silently summing raw
    # numbers across currencies with no FX conversion, and counting
    # failed/pending transactions as real spend, on any dataset shaped like
    # that. There's no exchange-rate data available to convert currencies,
    # and no reliable way to know which of an arbitrary source's status
    # strings means "settled" (vocabulary varies: Completed / Settled /
    # Success / Cleared / ...) - so the general, defensible fix is
    # majority-value filtering: restrict amount-aggregating queries to the
    # dataset's most common status and each entity's own most common
    # currency. In virtually all real transaction data the majority status
    # IS the settled one, and this keeps every aggregation internally
    # consistent (comparable units, real completed spend) instead of
    # silently wrong, at the cost of only covering the majority case.
    # Retrieval templates ("list my transactions") are left unfiltered -
    # they display raw rows, not a computed spend figure.
    majority_status = None
    if "status" in clean_df.columns:
        mode = clean_df["status"].mode()
        majority_status = mode.iloc[0] if not mode.empty else None

    for entity_id, grp in clean_df.groupby("entity_id"):
        dates = sorted(grp["timestamp"].tolist())
        categories = sorted(grp["category"].dropna().unique().tolist())
        windows = _month_windows(dates)
        if not windows:
            continue

        entity_currency = None
        if "currency" in grp.columns:
            cur_mode = grp["currency"].mode()
            entity_currency = cur_mode.iloc[0] if not cur_mode.empty else None

        attempts = 0
        produced = 0
        while produced < max_per_entity and attempts < max_per_entity * 4:
            attempts += 1
            kind, tpl_fn = rng.choice(TEMPLATES)
            built = tpl_fn(entity_id, categories, windows, rng)
            if built is None:
                continue
            variants, sql, params, result_kind = built

            if not has_negative_amounts:
                sql = sql.replace("amount<0", spend_filter)
                if tpl_fn is _tpl_largest_transaction:
                    sql = sql.replace("ORDER BY amount ASC", "ORDER BY amount DESC")
                elif tpl_fn is _tpl_smallest_transaction:
                    sql = sql.replace("ORDER BY amount DESC", "ORDER BY amount ASC")

            if kind != "retrieval":
                extra_filter, extra_params = "", []
                if majority_status is not None:
                    extra_filter += " AND status=?"
                    extra_params.append(majority_status)
                if entity_currency is not None:
                    extra_filter += " AND currency=?"
                    extra_params.append(entity_currency)
                if extra_filter:
                    # entity_id=? appears once per subquery - TWICE for
                    # _tpl_compare_periods, which has two independent nested
                    # SELECTs each with their own entity_id filter. Replace
                    # every occurrence (not just the first) and insert
                    # extra_params after every params entry equal to
                    # entity_id, so both the SQL text and the params tuple
                    # stay aligned regardless of how many subqueries there are.
                    sql = sql.replace("entity_id=?", "entity_id=?" + extra_filter)
                    new_params = []
                    for val in params:
                        new_params.append(val)
                        if val == entity_id:
                            new_params.extend(extra_params)
                    params = tuple(new_params)

            ok, result = run_query(conn, sql, params)
            if not ok:
                continue  # invalid SQL - never keep a query that doesn't execute
            if not _result_is_usable(result_kind, result["rows"]):
                continue  # executed fine but no meaningful answer - skip

            reference_answer = _format_reference_answer(result_kind, result["rows"], result["columns"])
            literal_sql = _render_literal_sql(sql, params)

            # split by the window's end date (or later window's end, for period comparisons)
            # NOTE: don't detect date params by substring-checking for "T" - a category
            # value like "Travel" also contains a capital T and would be misidentified
            # as a timestamp, crashing datetime.fromisoformat(). Parse-and-check instead.
            date_params = []
            for p in params:
                if isinstance(p, str):
                    try:
                        date_params.append(datetime.fromisoformat(p))
                    except ValueError:
                        pass
            window_end = max(date_params) if date_params else None
            split = "train"
            if window_end:
                if window_end <= train_cutoff:
                    split = "train"
                elif window_end <= val_cutoff:
                    split = "val"
                else:
                    split = "test"

            for variant in rng.sample(variants, min(phrasing_variants, len(variants))):
                examples.append({
                    "task": "query_sql",
                    "instruction": variant,
                    "output": literal_sql,  # directly executable - this is the real fine-tuning target
                    "reference_answer": reference_answer,
                    "template_kind": kind,
                    "split": split,
                    "_debug_sql_template": sql,   # parameterized form, for internal re-validation only
                    "_debug_sql_params": params,
                })
            produced += 1

    return examples

"""
Fact-grounded spend summarization (Option B from design discussion).

Rather than having the model read raw transactions and compute totals/trends
itself while generating prose (arithmetic-in-the-head, unreliable), every
fact in the narrative is computed by the SAME query engine used for the
NL-to-SQL task. The model's only job is phrasing already-correct numbers
into readable text - a "retrieve-then-generate" pattern, not memorization.
"""
import random
import calendar
from datetime import datetime, timedelta
from .query_engine import run_query

_CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹", "JPY": "¥"}


def _currency_symbol(code):
    # Falls back to "<CODE> " (e.g. "AUD 12.50") for a currency this
    # pipeline doesn't have a symbol for, rather than silently mislabeling
    # it as "$" - narratives previously hardcoded "$" regardless of the
    # entity's actual currency, so a EUR/GBP/INR spender's summary claimed
    # dollar amounts that were never really dollars.
    return _CURRENCY_SYMBOLS.get(code, f"{code} " if code else "$")


def _compute_facts(conn, entity_id, start: datetime, end: datetime,
                    spend_filter="amount<0", extra_where="", extra_params=(), currency=None):
    """All facts computed via real SQL aggregation - nothing guessed.

    spend_filter/extra_where/extra_params let the caller adapt to a source
    whose amounts aren't signed negative-for-spend, and restrict to a
    single currency/settled status - see generate_summarization_examples
    for why (same reasoning as nl_to_sql.py's identical adaptation).
    Hardcoding "amount<0" here meant EVERY fact computation returned None
    on an unsigned-amount source (SUM/COUNT with that filter matches
    nothing), so generate_summarization_examples produced zero examples
    end to end - confirmed on a real unsigned-amount test CSV.
    """
    facts = {"start": start, "end": end, "currency": currency}
    where_tail = f" AND {spend_filter}{extra_where} AND timestamp BETWEEN ? AND ?"

    ok, r = run_query(conn,
        "SELECT SUM(ABS(amount)), COUNT(*) FROM transactions WHERE entity_id=?" + where_tail,
        (entity_id, *extra_params, start.isoformat(), end.isoformat()))
    total, count = (r["rows"][0] if ok and r["rows"] else (None, 0))
    facts["total_spend"] = total
    facts["transaction_count"] = count or 0

    if not total or not count:
        return None  # nothing meaningful to summarize for this window

    ok, r = run_query(conn,
        "SELECT category, SUM(ABS(amount)) as t FROM transactions WHERE entity_id=?" + where_tail +
        " GROUP BY category ORDER BY t DESC LIMIT 1",
        (entity_id, *extra_params, start.isoformat(), end.isoformat()))
    if ok and r["rows"]:
        facts["top_category"], facts["top_category_amount"] = r["rows"][0]
        facts["top_category_pct"] = facts["top_category_amount"] / total * 100
    else:
        facts["top_category"] = None

    ok, r = run_query(conn,
        "SELECT counterparty, COUNT(*) as c FROM transactions WHERE entity_id=?" + where_tail +
        " GROUP BY counterparty ORDER BY c DESC LIMIT 1",
        (entity_id, *extra_params, start.isoformat(), end.isoformat()))
    if ok and r["rows"]:
        facts["top_merchant"], facts["top_merchant_count"] = r["rows"][0]
    else:
        facts["top_merchant"] = None

    # "largest single transaction" = most negative amount under the
    # negative-for-spend convention (ASC), but the largest POSITIVE amount
    # once spend_filter has switched to the unsigned convention (DESC) -
    # same direction-flip reasoning as nl_to_sql.py's largest/smallest.
    max_order = "ASC" if spend_filter == "amount<0" else "DESC"
    ok, r = run_query(conn,
        "SELECT amount, category, counterparty, timestamp FROM transactions WHERE entity_id=?" + where_tail +
        f" ORDER BY amount {max_order} LIMIT 1",
        (entity_id, *extra_params, start.isoformat(), end.isoformat()))
    if ok and r["rows"]:
        amt, cat, merch, ts = r["rows"][0]
        facts["max_amount"], facts["max_category"], facts["max_merchant"] = abs(amt), cat, merch

    # trend vs. the immediately preceding period of equal length
    period_len = end - start
    prev_start, prev_end = start - period_len, start - timedelta(seconds=1)
    ok, r = run_query(conn,
        "SELECT SUM(ABS(amount)) FROM transactions WHERE entity_id=?" + where_tail,
        (entity_id, *extra_params, prev_start.isoformat(), prev_end.isoformat()))
    prev_total = r["rows"][0][0] if ok and r["rows"] else None
    if prev_total:
        pct_change = (total - prev_total) / prev_total * 100
        facts["trend_direction"] = "up" if pct_change > 0 else "down"
        facts["trend_pct"] = abs(pct_change)
    else:
        facts["trend_direction"] = None

    return facts


def _render_narrative(facts, label, rng):
    total = facts["total_spend"]
    count = facts["transaction_count"]
    cur = _currency_symbol(facts.get("currency"))

    sentences = [f"In {label}, you made {count} transaction{'s' if count != 1 else ''} "
                 f"totaling {cur}{total:.2f}."]

    if facts.get("top_category"):
        sentences.append(
            rng.choice([
                f"Most of that went to {facts['top_category']} ({cur}{facts['top_category_amount']:.2f}, "
                f"{facts['top_category_pct']:.0f}% of your spending).",
                f"{facts['top_category']} was your top category at {cur}{facts['top_category_amount']:.2f} "
                f"({facts['top_category_pct']:.0f}% of the total).",
            ])
        )

    if facts.get("top_merchant"):
        sentences.append(
            rng.choice([
                f"You shopped most often at {facts['top_merchant']} ({facts['top_merchant_count']} times).",
                f"{facts['top_merchant']} was your most frequent merchant, with {facts['top_merchant_count']} visits.",
            ])
        )

    if facts.get("max_amount"):
        sentences.append(
            f"Your largest single transaction was {cur}{facts['max_amount']:.2f} "
            f"at {facts['max_merchant']} ({facts['max_category']})."
        )

    if facts.get("trend_direction"):
        sentences.append(
            f"Compared to the previous period, spending was {facts['trend_direction']} "
            f"by {facts['trend_pct']:.0f}%."
        )

    return " ".join(sentences)


def generate_summarization_examples(clean_df, conn, split_cutoffs, phrasing_variants=2, seed=0):
    """Fact-grounded summarization examples - every number in the output is
    traceable to a real SQL aggregation, only the phrasing varies."""
    rng = random.Random(seed)
    train_cutoff, val_cutoff = split_cutoffs
    examples = []

    # Same adaptation as nl_to_sql.py's generate_nl_sql_examples, for the
    # same reasons: an unsigned-amount source needs the sign filter
    # dropped (and max-amount direction flipped), and a source with mixed
    # currencies/non-settled statuses needs facts restricted to a single,
    # internally-consistent slice rather than silently mixing units.
    has_negative_amounts = (clean_df["amount"] < 0).any()
    spend_filter = "amount<0" if has_negative_amounts else "1=1"
    majority_status = None
    if "status" in clean_df.columns:
        mode = clean_df["status"].mode()
        majority_status = mode.iloc[0] if not mode.empty else None

    for entity_id, grp in clean_df.groupby("entity_id"):
        dates = sorted(grp["timestamp"].tolist())
        if len(dates) < 5:
            continue

        entity_currency = None
        if "currency" in grp.columns:
            cur_mode = grp["currency"].mode()
            entity_currency = cur_mode.iloc[0] if not cur_mode.empty else None
        extra_where, extra_params = "", []
        if majority_status is not None:
            extra_where += " AND status=?"
            extra_params.append(majority_status)
        if entity_currency is not None:
            extra_where += " AND currency=?"
            extra_params.append(entity_currency)

        seen_months = {}
        for d in dates:
            key = (d.year, d.month)
            if key not in seen_months:
                last_day = calendar.monthrange(d.year, d.month)[1]
                seen_months[key] = (
                    datetime(d.year, d.month, 1),
                    datetime(d.year, d.month, last_day, 23, 59, 59),
                    datetime(d.year, d.month, 1).strftime("%B %Y"),
                )

        for start, end, label in seen_months.values():
            facts = _compute_facts(conn, entity_id, start, end,
                                    spend_filter=spend_filter,
                                    extra_where=extra_where, extra_params=extra_params,
                                    currency=entity_currency)
            if facts is None:
                continue
            for _ in range(phrasing_variants):
                narrative = _render_narrative(facts, label, rng)
                split = "train"
                if end <= train_cutoff:
                    split = "train"
                elif end <= val_cutoff:
                    split = "val"
                else:
                    split = "test"
                examples.append({
                    "task": "spend_summarization",
                    "instruction": f"Summarize my spending in {label}.",
                    "output": narrative,
                    "facts_used": {k: v for k, v in facts.items() if k not in ("start", "end")},
                    "output_is_fact_grounded": True,
                    "split": split,
                })

    return examples

"""
Synthetic transaction data generator.

Produces two DIFFERENT raw schemas (as if from two different providers) so the
demo can show schema resolution actually resolving ambiguity, not just reading
a pre-matched CSV:

  Provider A ("standard-ish"): readable column names, signed dollar amounts,
                                ISO dates.
  Provider B ("messy"):        abbreviated column names, integer amounts in
                                CENTS, ambiguous DD/MM date strings, sign given
                                via a separate debit/credit column.

Both encode the same underlying synthetic behavior, including a small
injected fraud rate, so downstream task shaping has something real to work with.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

CATEGORIES = ["groceries", "dining", "travel", "electronics", "utilities",
              "entertainment", "subscription", "healthcare", "fuel", "clothing"]
CHANNELS = ["card_present", "card_not_present", "online", "atm", "mobile_app"]
MERCHANTS = ["Whole Foods", "Amazon", "Uber", "Delta Airlines", "Netflix",
             "Shell", "Target", "Best Buy", "CVS Pharmacy", "Starbucks",
             "Spotify", "City Power & Light"]

# Merchants realistically imply a category most of the time - this gives
# the categorization task genuine learnable signal (a small noise rate
# keeps it non-trivial rather than a lookup table).
# NOTE: every one of the 10 CATEGORIES needs at least one dedicated merchant
# here - two categories (subscription, utilities) previously had none,
# meaning they only ever appeared via the 10% noise injection below and
# were ~10x underrepresented. This was found via real per-class accuracy
# results (both the from-scratch model and a real fine-tuned DistilBERT
# independently failed on exactly these two categories, at 0%, every run).
MERCHANT_CATEGORY = {
    "Whole Foods": "groceries", "Amazon": "electronics", "Uber": "travel",
    "Delta Airlines": "travel", "Netflix": "entertainment", "Shell": "fuel",
    "Target": "clothing", "Best Buy": "electronics", "CVS Pharmacy": "healthcare",
    "Starbucks": "dining", "Spotify": "subscription", "City Power & Light": "utilities",
}
CATEGORY_NOISE_RATE = 0.10  # fraction of rows where category doesn't match the merchant's usual one


def _generate_base(n_entities=40, days=90, txns_per_entity=(15, 60)):
    rows = []
    start_date = pd.Timestamp("2026-01-01")

    for entity_idx in range(n_entities):
        entity_id = f"CUST{entity_idx:04d}"
        n_txns = RNG.integers(*txns_per_entity)
        offsets = np.sort(RNG.uniform(0, days, size=n_txns))

        for offset in offsets:
            ts = start_date + pd.Timedelta(days=float(offset), hours=float(RNG.integers(0, 24)))
            merchant = RNG.choice(MERCHANTS)
            category = MERCHANT_CATEGORY[merchant] if RNG.random() > CATEGORY_NOISE_RATE else RNG.choice(CATEGORIES)
            channel = RNG.choice(CHANNELS)
            amount = round(float(RNG.lognormal(mean=3.2, sigma=0.9)), 2)
            is_fraud = 0

            # inject rare anomalies: unusually large amount, odd hour, online channel
            if RNG.random() < 0.02:
                amount = round(amount * RNG.uniform(6, 15), 2)
                channel = "online"
                is_fraud = 1

            rows.append({
                "entity_id": entity_id,
                "timestamp": ts,
                "amount": amount,
                "currency": "USD",
                "category": category,
                "channel": channel,
                "counterparty": merchant,
                "status": "COMPLETED",
                "label": is_fraud,
            })
    return pd.DataFrame(rows)


def generate_provider_a():
    """Standard-ish schema: readable names, signed dollars, ISO dates."""
    df = _generate_base()
    out = pd.DataFrame({
        "account_id": df["entity_id"],
        "txn_date": df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "amount": -df["amount"],  # spend as negative, standard signed convention
        "currency": df["currency"],
        "merchant_category": df["category"],
        "payment_channel": df["channel"],
        "merchant_name": df["counterparty"],
        "txn_status": df["status"],
        "is_fraud": df["label"],
    })
    return out


def generate_provider_b():
    """Messy schema: abbreviated names, integer cents, DD/MM dates, dr/cr column."""
    df = _generate_base()
    out = pd.DataFrame({
        "acct_no": df["entity_id"],
        "date": df["timestamp"].dt.strftime("%d/%m/%Y %H:%M"),
        "amt_cents": (df["amount"] * 100).round().astype(int),
        "mcc": df["category"],
        "chnl": df["channel"],
        "merchant": df["counterparty"],
        "dr_cr": "DR",  # all spend transactions are debits in this synthetic set
        "fraud_flag": df["label"],
    })
    return out


if __name__ == "__main__":
    generate_provider_a().to_csv("provider_a_sample.csv", index=False)
    generate_provider_b().to_csv("provider_b_sample.csv", index=False)
    print("Wrote provider_a_sample.csv and provider_b_sample.csv")

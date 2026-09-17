"""
Canonical transaction schema and alias dictionary.
This is the single source of truth for what a "clean" transaction looks like,
regardless of which raw source it came from.
"""

CANONICAL_FIELDS = {
    "entity_id":    {"required": True,  "kind": "id"},
    "timestamp":    {"required": True,  "kind": "datetime"},
    "amount":       {"required": True,  "kind": "numeric"},
    "currency":     {"required": False, "kind": "categorical", "default": "USD"},
    "category":     {"required": False, "kind": "categorical", "default": "UNKNOWN"},
    "channel":      {"required": False, "kind": "categorical", "default": "UNKNOWN"},
    "counterparty": {"required": False, "kind": "id",          "default": "UNKNOWN"},
    "status":       {"required": False, "kind": "categorical", "default": "COMPLETED"},
    "label":        {"required": False, "kind": "categorical", "default": None},
}

REQUIRED_FIELDS = [f for f, spec in CANONICAL_FIELDS.items() if spec["required"]]

# Known header aliases -> canonical field. Matched after normalization
# (lowercase, strip non-alphanumeric).
ALIAS_DICTIONARY = {
    "entity_id": [
        "entityid", "accountid", "customerid", "userid", "acctid",
        "custno", "accountnumber", "cardid", "acctno",
    ],
    "timestamp": [
        "timestamp", "date", "txndate", "transactiondate", "postedat",
        "datetime", "time", "createdat",
    ],
    "amount": [
        "amount", "amt", "transactionamount", "value", "txnamount",
        "total", "amtcents",
    ],
    "currency": ["currency", "curr", "ccy"],
    "category": [
        "category", "merchantcategory", "mcc", "txncategory", "typecategory",
    ],
    "channel": ["channel", "txnchannel", "paymentchannel", "method", "chnl"],
    "counterparty": [
        "counterparty", "merchant", "receiver", "payee", "recipient", "merchantname",
    ],
    "status": ["status", "txnstatus", "state"],
    "label": ["label", "isfraud", "fraudflag", "fraud", "target", "class", "fraud_flag"],
    # not a canonical field itself, but recognized so it can drive sign resolution
    "_direction": ["drcr", "direction", "debitcredit", "txntypesign", "dr_cr"],
}

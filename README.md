# Universal Transaction Preprocessing Pipeline

A modular, config-driven pipeline that turns arbitrary raw transaction CSVs into
clean, task-shaped, LLM-fine-tuning-ready output — with a Streamlit UI that
walks through each stage for demos/reviews.

## Structure

```
txn_pipeline/
├── app.py                      # Streamlit UI (entry point for the demo)
├── data_synth.py                # Synthetic data generator (2 different raw schemas)
├── pipeline/
│   ├── schema.py                 # Canonical schema + alias dictionary
│   ├── schema_resolution.py      # Stage 0: maps any CSV -> canonical schema
│   ├── cleaning.py               # Stage A: validation, PII hashing, causal features
│   ├── shaping.py                # Stage B: task-aware LLM output shaping
│   ├── manifest.py               # Audit trail builder
│   └── orchestrator.py           # Runs all stages, single entry point
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the demo UI

```bash
streamlit run app.py
```

Opens in your browser. Pick a synthetic data source and task type(s) in the
sidebar, click **Run Pipeline**, and step through the tabs to see schema
resolution, cleaning, and task-shaped output — with download buttons for the
JSONL output and the run manifest.

## Using the pipeline on your own data (no UI)

```python
import pandas as pd
from pipeline import run_pipeline

df = pd.read_csv("your_transactions.csv")
result = run_pipeline(df, source_name="my_source", task_types=["fraud_flagging"])

if result["fatal_error"]:
    print("Pipeline halted:", result["fatal_error"])
else:
    examples = result["shaped_outputs"]["fraud_flagging"]
    manifest = result["manifest"].to_dict()
```

## What's automated vs. what needs input

See the accompanying blueprint document for the full design rationale. In short:
- Schema mapping, unit/convention detection, PII hashing, causal feature
  engineering, time-based splitting, and output serialization are fully automated.
- Task type, PII policy, and (for tasks needing labels not present in raw
  data) a label source are required upfront inputs — the pipeline flags
  rather than fabricates these.
- Every automated decision is logged in the manifest for audit/review.

## Time-based train/val/test splitting

Transactions are split chronologically, never randomly: the earliest slice
trains, the middle slice validates, the most recent slice tests — matching
how the model will actually be used (never evaluated on data "from the
future" relative to what it trained on). Split fractions are configurable
in the sidebar. A held-out test split is included by default for fraud
flagging / categorization (clean, measurable metrics), and skipped by
default for summarization / general behavioral (mostly qualitative
evaluation) — toggleable either way. Each shaped example carries a `split`
field based on its target/most-recent transaction's timestamp, and
downloads are split-aware (separate train/val/test JSONL files per task).

## Stage C: training sanity check

A genuine (if small) transformer, implemented from scratch in plain NumPy
(no PyTorch/TensorFlow, no internet access, no pretrained weights), trains
directly on the pipeline's own categorization output to check whether the
shaped data has real learnable signal - not just plausible-looking text.
Shows a loss curve, validation accuracy vs. a random-guess baseline, and
before/after sample predictions. This is a **data sanity check**, not a
production training system - it answers "does this data have structure a
model could learn from," not "how will a full LLM fine-tune perform."
The backward pass (gradients) has been verified against numerical
gradient checking for correctness.

## Known limitations (by design, not oversight)

- **Spend summarization** targets are rule-based placeholders in this demo —
  real fine-tuning needs human- or LLM-labeled ground-truth summaries.
- **Fraud flagging / categorization** require a label column in the source
  data (or a joinable label file) — the pipeline will not fabricate labels.
- Schema resolution flags low-confidence mappings rather than guessing; very
  unusual or ambiguous CSVs may need a one-time manual mapping override.
  Resolution is confidence-ordered across ALL columns (a two-pass process:
  strong name-matches claim their field first, weaker content-only guesses
  fill in leftover fields second) — this specifically prevents a weak,
  order-dependent guess (e.g. a transaction ID column looking vaguely
  "ID-like") from stealing a required field's slot ahead of the column
  that's actually the correct, exact-name match for it.
- **Missing entity_id column**: if no account/customer identifier exists in
  the source file, the UI offers "auto-index rows" as an explicit opt-in
  fallback (each row becomes its own entity). This is clearly flagged in the
  manifest because it collapses entity-level sequencing to length 1 —
  fraud flagging, spend summarization, and general behavioral tasks lose
  most of their signal under this fallback, while categorization is
  unaffected (it doesn't depend on entity grouping). Always prefer a real
  identifier column when one is available.

## Quarantined columns are not "judged useless"

A column being quarantined means it didn't fit the fixed canonical schema —
it is NOT a claim that the column has no value for training. Columns like
transaction type, risk flags, or error codes can be highly informative
(especially for fraud flagging) despite not mapping to any of the 9
canonical fields. The Stage 0 tab lets you explicitly opt in to carry
specific quarantined columns through as extra context in the shaped
output. This is opt-in and clearly warned because these columns bypass
the pipeline's PII-hashing and validation entirely — only select columns
you've already reviewed.

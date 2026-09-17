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

## v5: mapping trust & training diagnostics

**Schema mapping now covers `category` and `label`, not just required fields**
— these can silently default (to `UNKNOWN` / no label) if unmapped, which
can quietly empty out the categorization/fraud-flagging task output with
no warning. The mapping form now offers to map these too, alongside a
plain-language summary ("Confident about: X. Worth double-checking: Y")
and a manual override tool for correcting ANY field mapping — including
ones that already auto-mapped, in case automation got it wrong.

**Training sanity check now includes real diagnostics, not just a loss
curve:**
- Majority-class baseline alongside random-guess baseline — a model that
  always predicts the most common category can look good against random
  chance while having learned nothing; the majority baseline catches that.
- Train accuracy tracked alongside validation accuracy, to surface
  overfitting (train climbing while val stalls/drops).
- Per-class accuracy breakdown — an overall good score can hide a category
  the model gets wrong 0% of the time.
- Small-sample warning when train/val counts are low enough that results
  are more noise than signal.
- An explicit verdict (✅ learned real signal / ⚠️ weak signal / ❌ no
  signal) instead of leaving the raw numbers to be eyeballed.

## v6: Natural-language querying (NL-to-SQL) and fact-grounded summarization

**New task: `query_sql`** — teaches a model to translate natural-language
questions ("How much did I spend on groceries in March?", "Show me my
transactions between these dates") into SQL, rather than teaching it to
compute answers directly. This avoids two real problems with direct
answer-memorization: (1) an LLM doing arithmetic token-by-token is
unreliable, and (2) a memorized number only holds for the exact data it
was trained on — it doesn't generalize to new transactions. Instead:

- A real SQLite database is built from the cleaned transaction data
  (`pipeline/query_engine.py`).
- Questions are generated combinatorially from template FAMILIES
  (aggregation, extremes, comparison, retrieval) instantiated against
  whatever categories/time-windows actually exist in your data — not a
  fixed universal question list.
- **Every generated SQL query is executed against the real data and
  validated before being kept.** A query that errors or returns nothing
  meaningful is dropped, not included as a bad example.
- 3 phrasing variants per question, so the model isn't only exposed to one
  rigid sentence structure per query shape.

**`spend_summarization` rewritten to be fact-grounded, not placeholder-based**
— every number in the narrative (total spend, top category, top merchant,
largest transaction, trend vs. previous period) is computed via real SQL
aggregation through the same query engine. The model's only job is
phrasing already-correct facts into readable prose — a "retrieve-then-generate"
pattern, not memorization. This replaces the old rule-based placeholder
that was explicitly flagged as fake ground truth.

**Honest limitations of this approach:**
- Query scope is "my spending" (per-entity) — not cross-entity/analyst
  aggregate queries.
- Template coverage, while combinatorial, is still finite — a sufficiently
  unusual real-world question may fall outside what was generated.
- At inference time, this design requires the actual query engine to be
  available to execute the model's generated SQL — the model alone
  doesn't produce a final answer, only the query.

## v7: bug fix - overriding an already-mapped field caused a crash

Using the "🔧 Override a field mapping" tool (added in v5) on a field that
was already auto-mapped from a different column (e.g. re-pointing `channel`
away from its auto-detected column) left BOTH the old and new source
columns mapped to the same canonical field. This produced two columns
named identically after renaming, which crashed downstream with
`sqlite3.OperationalError: duplicate column name` when building the query
engine for query_sql/spend_summarization.

Fixed at the root: applying an override now explicitly supersedes (removes
and re-quarantines) whichever column previously held that canonical field.
Also added a defensive check in the cleaning step that fails with a clear,
specific error message if this class of conflict ever recurs, instead of
surfacing a cryptic error several layers downstream.

## v10: bugs found via adversarial stress-test data

A deliberately messy, independently-designed stress-test CSV (not shaped to
match the pipeline's known handling) surfaced two real crashes and one
silent-wrong-data bug, all now fixed:

1. **Crash: mixed timezone-aware and naive timestamps** (e.g. a column mixing
   `2026-01-05T14:30:00Z` with plain `2026-01-05`) caused an unhandled
   `ValueError` from pandas, taking down the whole pipeline run. Fixed with a
   shared `safe_to_datetime()` helper that retries with UTC normalization
   instead of crashing.
2. **Crash: a fragile date-parameter detection heuristic** in the NL-to-SQL
   task checked for `"T"` as a substring to identify date strings - which
   incorrectly matched category values like `"Travel"` (capital T) and
   crashed `datetime.fromisoformat()`. Fixed by actually attempting to parse
   each string and checking whether it succeeds, rather than guessing from
   content.
3. **Silent wrong data: European-format amounts** like `"1.234,56"` (meaning
   1234.56) were being parsed as `1.23456` - a wrong number, not a dropped
   row, which is the worse failure mode. Fixed with a proper amount
   normalizer that distinguishes European (comma decimal) from US (period
   decimal) formatting by checking which separator appears last in the
   string, and also fixed a few other previously-dropped-but-fixable amount
   formats: trailing-minus notation (`123.45-`), currency codes as
   prefix/suffix (`USD 123.45`, `45.00 USD`), and additional currency
   symbols (₹, ¥).
4. **A single malformed CSV row (wrong field count, stray delimiter) used to
   reject the entire file upload**, even when every other row was fine.
   The upload path now retries with lenient parsing (skipping only the
   unparseable rows) and tells the user exactly what was skipped and why,
   instead of rejecting the whole file on one bad line.

Known, documented (not auto-fixed) limitations found during this same test:
- Excel serial dates and Unix epoch timestamps in a plain numeric column are
  not auto-detected - they're correctly left as unparsed rather than
  guessed, but this means Excel exports with numeric-formatted date columns
  will need those columns explicitly reviewed.
- Fully ambiguous all-numeric dates (e.g. `26-01-05`, three two-digit groups
  with no separator convention clues) can resolve inconsistently depending
  on value order. This is a genuine, largely unfixable ambiguity in the
  source data itself, not a pipeline defect - flagged here for awareness.

See `messy_transactions_stress_test.csv` for the exact adversarial dataset
used to find these issues - it's kept as a regression fixture.

## v11: fixed a second, different cause of the "duplicate column name" crash

The v7 fix only covered duplicates caused by the manual field-override tool.
This is a genuinely different root cause: if a dataframe with truly
duplicate-labeled columns for a mapped field ever reaches the cleaning
step (confirmed reproducible, though the exact real-world trigger in a
user's file wasn't pinned down), the schema-mapping dict itself stays
clean (dict keys can't duplicate) but the resulting dataframe can still end
up with two identically-named columns - bypassing the v7 check entirely,
since that check only inspects the mapping dictionary, not the actual
column index of the result.

Fixed with defense in depth at two points:
1. `apply_schema_mapping` now checks the ACTUAL resulting columns (not just
   the mapping dictionary) for duplicates immediately after mapping, keeps
   the first occurrence, and logs a clear manifest warning if this happens.
2. `build_sqlite_db` has the same defensive check as a last line of
   defense, in case a duplicate ever reaches that point through a path we
   haven't traced yet.

Both paths now degrade gracefully (keep first occurrence, warn loudly) 
instead of crashing. If you see the warning "Duplicate column name(s) after
schema mapping" in a manifest, please share the source file's column
headers - that will help pin down the exact trigger for a permanent fix at
the true root cause, rather than just the safety net now in place.

## v12: fixed a confusing silent collision (channel vs payment_method)

Removed `payment_method` from the `channel` alias list - it's a genuinely
different concept (payment instrument vs. transaction location/method) that
can coexist with a real `channel` column in the same file, and treating
them as synonyms caused a real, confusing bug: when both columns were
present, `payment_method` sometimes won the `channel` slot, and the
literal `channel` column was silently quarantined with no explanation.

More importantly, added a general safeguard so this CLASS of problem can
never be silent again, regardless of which alias causes it: whenever two
columns both achieve a strong name-based match for the same canonical
field, the losing column is now marked with a new status
(`quarantined_collision`, shown as 🟠 in the Stage 0 table) and a specific
note explaining exactly what happened and pointing to the field-override
tool - instead of disappearing into the generic quarantined bucket with no
explanation.

## v13: fixed COUNT examples showing "0.00" (formatting bug + a confusing-example bug)

Two separate issues, both real:

1. **Formatting**: COUNT query results were formatted with the same
   `f"{val:.2f}"` used for SUM/AVG, so a count of 12 displayed as "12.00" -
   a transaction count is a whole number and should never show decimal
   places. Fixed by giving COUNT its own result kind, formatted as a plain
   integer.
2. **A genuinely confusing training example**: time windows for NL-to-SQL
   examples are built from ALL of an entity's transaction dates, but the
   COUNT/SUM templates only count spend (`amount<0`). If an entity's only
   activity in a given month was a refund (a positive amount), a window
   still got generated for that month, and "how many transactions did I
   make" would correctly-by-scope but confusingly answer "0" - even though
   a transaction (the refund) did happen that month. Fixed by requiring
   COUNT examples to have a genuine positive count before being kept,
   rather than allowing a technically-consistent-but-misleading zero
   through.

Both were found and confirmed via direct reproduction (a synthetic
refund-only month) before fixing, and verified against real pipeline
output afterward.

## v14: cross-dataset / cross-model evaluation harness (+ a real pipeline bug fix)

New `eval_harness/` module and `evaluation_notebook.ipynb` (Colab-ready):

1. **`evaluation_harness.py`** - runs multiple datasets x multiple sizes of the
   from-scratch tiny transformer x multiple random seeds, producing one
   comparable results table and a heatmap. Fully tested here (no GPU needed).
   A real finding from testing this: "medium" model config underperformed
   "small" on one dataset with high seed variance - a genuine illustration
   of why multiple seeds matter before trusting a single comparison.
2. **`external_llm_eval.py`** - fine-tunes REAL pretrained models on GPU
   (DistilBERT for categorization, DistilGPT-2 + LoRA for NL-to-SQL). The
   SQL evaluation executes generated queries against the same SQLite engine
   the pipeline uses and checks the result matches - a real correctness
   metric, not text similarity. **Not execution-verified in this sandbox**
   (no GPU here, and disk constraints prevented even a CPU-only structural
   test) - written carefully against stable HF APIs, but flagged honestly
   as needing its first real run in Colab.
3. **`evaluation_notebook.ipynb`** - ties both together with setup, GPU
   installation, and a combined from-scratch-vs-real-model comparison table.

**A real pipeline bug found while building this**: `query_sql` examples were
storing SQL with `?` placeholders as the fine-tuning target - fine for the
pipeline's own internal validation (where params are supplied separately),
but **unusable at actual inference time**, since nothing would exist to
bind those placeholders to real values. Fixed by rendering the literal,
directly-executable SQL (with real quoted values substituted in) as the
actual `output` field going forward - verified that all 1,440 examples
across both synthetic providers execute correctly standalone with zero
external parameters.

## v15: fixed the notebook's setup cell silently "succeeding" on failure

A user hit "cannot find or open txn_preprocessing_pipeline_v14.zip" followed
by several cascading errors, but the cell still printed "Setup complete" at
the end - because shell commands (`!unzip`, `!pip install`) don't stop
notebook execution on failure the way Python code does, so the final print
ran regardless of whether anything upstream actually worked.

Fixed by rewriting the setup cell to explicitly check the result of every
step (zip found, unzip succeeded, folder exists, requirements.txt exists,
pip install succeeded) and raise a clear, specific error immediately at
whichever step actually failed - verified by extracting the real cell code
and running it standalone to confirm it raises correctly rather than just
printing diagnostics. Also added an automatic upload prompt if no matching
zip is found in the session at all (Colab sessions are ephemeral - a
runtime reset silently deletes previously uploaded files, which is a
likely cause of "file not found" errors like this one).

## v16: fixed synthetic data category imbalance (found via real Colab evaluation)

Real evidence from the first actual Colab run of the DistilBERT categorization
eval: per-class accuracy showed `subscription` and `utilities` at exactly
0.0% across all epochs - identical failure to what the from-scratch tiny
transformer had shown earlier, on the exact same two categories.

Root cause: of 10 CATEGORIES, only 8 had a dedicated merchant in
MERCHANT_CATEGORY - "travel" and "electronics" each had two merchants,
leaving "subscription" and "utilities" with none. They only ever appeared
via the 10% random noise injection, making them ~10x underrepresented.
No amount of fine-tuning can learn a category that's barely in the data.

Fixed by adding two new merchants ("Spotify" -> subscription,
"City Power & Light" -> utilities) so all 10 categories have dedicated
representation. Verified the category distribution is now balanced
(~90-211 examples per category, vs. near-zero for the two affected
categories before).

**Important, expected side effect**: the from-scratch tiny transformer's
accuracy on categorization dropped substantially after this fix (e.g. from
~76-92% down to a much more variable 8-40% depending on seed/model size).
This is NOT a regression - the earlier numbers were inflated by the same
bug, since the model could ignore 2 nearly-absent categories almost for
free. The corrected, properly-balanced 10-class task is genuinely harder,
and this drop reveals the tiny from-scratch model's real capacity limit at
this task's true difficulty - which in turn makes a real pretrained model's
performance (e.g. DistilBERT) a much more meaningful comparison than
before. Re-run any Colab evaluations against this version to get honest,
non-inflated numbers.

## v17: fixed the NL-to-SQL fine-tuning "runaway generation" bug (found via real Colab run)

Real Colab evidence at epochs=15: the model had genuinely learned correct
SQL structure and columns, but every single generated query had the exact
same malformed tail appended regardless of the question - e.g.
`... LIMIT 1 AND 2 ORDER BY amount DESC` glued onto an otherwise-complete,
correct query. A "doesn't know when to stop" bug, not a comprehension bug.

Root cause: training loss was computed over the ENTIRE tokenized sequence
(question text + SQL answer + padding), not just the answer - standard
practice for instruction-style fine-tuning is to mask the prompt so the
model is only ever graded on generating the answer. Without that masking,
the signal for "where does the answer end" was diluted, so the model never
reliably learned to stop.

Fixed with three changes to `external_llm_eval.py`'s `run_sql_generation_eval`:
1. Proper prompt-loss masking - labels are now -100 (ignored) for every
   prompt/padding token, real token ids only for the SQL answer + EOS.
   Verified the masking logic's control flow directly with a mock tokenizer
   before shipping (real GPT-2 tokenizer/torch execution still needs Colab).
2. Explicit `eos_token_id` passed to `generate()`, not just relying on
   defaults.
3. Added `repetition_penalty=1.3` and `no_repeat_ngram_size=3` to directly
   prevent the kind of repeat-loop degenerate output seen in earlier testing.
4. Widened LoRA target modules from `["c_attn"]` to `["c_attn", "c_proj"]`
   for more adaptation capacity.

This was diagnosed through several rounds of real Colab execution and
output inspection - each round narrowed the actual root cause rather than
guessing, the same iterative process used throughout this project.

## v18: all 5 tasks now have a real-model eval path; NL-to-SQL termination fix v2

**Termination fix, round 2** (real Colab evidence from v17 showed partial
improvement - different garbage, same core problem: still not stopping
reliably). Added an explicit, distinctive stop signal the model doesn't
have to infer from EOS alone: every training answer now ends with `;`
before EOS - a single consistent character is a much easier pattern for a
small model to learn than picking EOS out of the entire vocabulary. Also
added a code-level safety net regardless of the model's own behavior:
generated text is truncated at the first `;` before execution. Reduced
`max_new_tokens` from 80 to 45 (SQL answers here are short one-liners) to
shrink the space available for runaway generation. Verified sqlite3
tolerates a trailing semicolon correctly before shipping this.

**Real-model coverage extended to all 5 pipeline tasks** (previously only
categorization + query_sql had one):
- `run_classification_eval()` generalized (one-line change: join the full
  `context` list, not just `context[0]`) to work directly for
  **fraud_flagging** too, since it's structurally the same job as
  categorization - no new function needed.
- New `run_summarization_eval()` for **spend_summarization** - since
  summaries are open-ended prose with no execution check available,
  correctness is measured as a fact-inclusion rate (does the generated
  text actually contain the real computed facts it was supposed to
  report). This scoring logic was unit-tested directly in this sandbox
  (pure Python, no GPU needed) and correctly distinguishes a summary that
  reports real facts (score 1.0) from a vague one that doesn't (score 0.0).
- New `run_general_behavioral_eval()` for **general_behavioral** - no
  correctness check applies here either (it's continued-pretraining-style
  sequence modeling), so evaluation uses standard held-out perplexity.

Notebook updated: Part 1 now requests all 5 task types (fixes the earlier
`KeyError: 'query_sql'` gap directly at the source rather than relying on
the user remembering to edit it), and Parts 3c/3d/3e + an expanded Part 4
comparison table cover the 3 newly-added tasks.

**Same honesty note as before applies to the 2 new functions**: the
fact-inclusion scoring logic was verified directly; the actual GPU
training/generation code around it has not been execution-tested in this
sandbox (no GPU, disk constraints prevent even a CPU-only structural
test) - your next Colab run is the first real test of these two.

## v18: all 5 tasks now have a real-model evaluation path (+ the SQL fix confirmed applied)

Extended `external_llm_eval.py` and the notebook to cover the 3 tasks that
previously had no real-model eval function:

- **Fraud flagging** - reuses `run_classification_eval` directly (same
  underlying job as categorization: text in, label out), after
  generalizing that function to join a full multi-transaction context
  window instead of assuming a single transaction.
- **Spend summarization** - new `run_summarization_eval`. Since summaries
  are open-ended prose with no execution check possible, correctness is
  measured as a fact-inclusion rate: does the generated text actually
  contain the real computed facts (total spend, top category, top
  merchant, max transaction) - verified the scoring logic directly
  (perfect score for a summary stating the real numbers, zero for vague
  text with no numbers).
- **General behavioral** - new `run_general_behavioral_eval`. This task has
  no instruction/answer pairs by design, so "accuracy" doesn't apply -
  evaluated via held-out perplexity instead (lower = the model finds real
  transaction sequences more predictable).

Also confirms the semicolon-termination fix from the previous round (fixing
the SQL runaway-generation bug) is in place: training now appends a
distinctive `;` stop marker before EOS, and generation truncates at the
first `;` as a code-level safety net regardless of the model's own
stopping behavior.

Notebook Part 1 now requests all 5 task types by default, and Part 4's
comparison table includes all 5 real-model results alongside the
from-scratch baseline in one view.

## v19: fixed the torchvision/VideoReader crash in Colab (real, reproducible)

Confirmed via two identical Colab tracebacks: `run_sql_generation_eval`,
`run_summarization_eval`, and `run_general_behavioral_eval` all crashed
with `ImportError: cannot import name 'VideoReader' from 'torchvision.io'`
the moment training started - before any of our own logic even ran.

Root cause: all three functions called `.set_format(type="torch", ...)`
on the Dataset before training. This routes tensor conversion through
`datasets`' own torch formatter, which - completely unrelated to our
text-only task - does a `torchvision.io.VideoReader` availability check to
support video datasets. Colab's pre-installed torchvision version doesn't
expose `VideoReader` where `datasets` expects it, so the import crashes
outright, for every user, unconditionally - not an environment quirk
specific to one session.

Fixed at the root by removing all three `.set_format(...)` calls. Features
are left as plain Python lists; `Trainer`'s default collator builds
tensors directly via `torch.tensor()`, which never touches the broken
`datasets`/`torchvision` code path at all. This removes the need for the
`!pip uninstall torchvision` workaround entirely - training should now
work in a fresh Colab session with no extra steps.

## v20: fixed SQL generation params corrupting valid SQL + added class-weighted loss for imbalanced classification

Two real fixes from continued Colab diagnosis:

1. **`run_sql_generation_eval`**: `repetition_penalty=1.3` and
   `no_repeat_ngram_size=3` (added to fix the earlier runaway-repetition
   bug) were confirmed via direct A/B testing to actively corrupt valid
   SQL - a correct `BETWEEN 'date1' AND 'date2'` clause legitimately needs
   to repeat similar tokens (dates, `AND`), and penalizing repetition
   forced the model off the correct continuation at exactly that point,
   every time. Removed both settings; the semicolon-truncation safety net
   (added previously) still guards against genuine runaway generation
   without fighting valid SQL structure. `max_new_tokens` raised 45 -> 60
   since a real generated date was seen getting cut off mid-token.

2. **`run_classification_eval`**: added `use_class_weights` (default True).
   Real Colab evidence: fraud flagging hit 96.72% accuracy - identical to
   the majority baseline - because the model learned to always predict
   LEGITIMATE (per-class accuracy: FRAUD 0.0, LEGITIMATE 1.0). With severe
   class imbalance, a model can match the majority baseline while never
   learning the rare class at all. Fixed with inverse-frequency class
   weighting in the loss (verified the weighting math directly: at the
   real ~97/3 split observed, FRAUD is weighted ~29x higher than
   LEGITIMATE, via a custom `WeightedLossTrainer` subclass). This is a
   general option, not fraud-specific - any imbalanced classification task
   run through this function benefits from it.

Both fixes were reached through multiple rounds of real Colab execution,
targeted diagnostics, and confirmation before changing code - the same
process used throughout this project.

## v21: fixed general_behavioral producing ZERO training examples (real bug, found via Colab crash)

Real Colab evidence: `run_general_behavioral_eval` printed "Training on 0
sequences..." then crashed with a confusing column-signature error - which
was just a downstream symptom of training on a genuinely empty dataset.

Root cause, confirmed directly: `build_general_examples` took only each
entity's MOST RECENT `max_context` transactions (`.tail(max_context)`),
producing exactly ONE window per entity, and tagged that window's split
using its last transaction's timestamp. Since every entity's "most recent
activity" clusters near the end of the WHOLE dataset's time range by
definition, and splits are time-based (earliest trains, latest tests),
this structurally guaranteed nearly every example would land in `test` -
confirmed: 100% of examples (40/40) were in `test`, with `train` and `val`
completely empty, for both synthetic providers.

Fixed in `pipeline/shaping.py`: `build_general_examples` now generates
multiple overlapping sliding windows across each entity's FULL history
(50% overlap, similar in spirit to how fraud_flagging already samples
multiple windows per entity), instead of a single tail-end window. This
naturally spreads examples across the whole timeline, giving genuine
train/val/test coverage - verified directly: Provider A went from
40 examples (100% test, 0% train/val) to 290 examples properly split
train=182/val=42/test=66. Also incidentally increases the amount of
general_behavioral training data substantially.

This is a core pipeline fix (not eval-harness-specific) - affects anyone
using the general_behavioral task, not just the Colab evaluation notebook.

"""
Same as run_all_fixed.py, but against the real diverse_transactions.csv
instead of synthetic Provider A/B - the real test of whether tonight's
pipeline fixes (sign-convention detection, currency/status filtering,
currency-symbol correctness) actually translate into learnable, correct
examples on real-world data, not just higher example counts.
"""
import sys
sys.path.insert(0, "/Users/ranayilmaz/Desktop/txn_pipeline")
sys.path.insert(0, "/Users/ranayilmaz/Desktop/txn_pipeline/eval_harness")

import pandas as pd
from pipeline.orchestrator import run_pipeline
from pipeline.query_engine import build_sqlite_db
from external_llm_eval import run_classification_eval, run_sql_generation_eval, run_summarization_eval

ALL_TASKS = ["categorization", "fraud_flagging", "query_sql", "spend_summarization"]

raw = pd.read_csv("/Users/ranayilmaz/Desktop/diverse_transactions.csv")
result = run_pipeline(raw, "diverse_transactions", ALL_TASKS)
if result["fatal_error"]:
    raise SystemExit(f"fatal_error: {result['fatal_error']}")

train_by_task, val_by_task = {}, {}
for t in ALL_TASKS:
    examples = result["shaped_outputs"][t]
    train_by_task[t] = [e for e in examples if e["split"] == "train"]
    val_by_task[t] = [e for e in examples if e["split"] in ("val", "test")]
    print(f"{t}: train={len(train_by_task[t])} val={len(val_by_task[t])}")

print("\n" + "=" * 70)
print("1/4 CATEGORIZATION (DistilBERT)")
print("=" * 70)
cat_result = run_classification_eval(train_by_task["categorization"], val_by_task["categorization"], epochs=3)

print("\n" + "=" * 70)
print("2/4 FRAUD FLAGGING (DistilBERT, oversampling + class-weighting)")
print("=" * 70)
fraud_result = run_classification_eval(train_by_task["fraud_flagging"], val_by_task["fraud_flagging"], epochs=4)

print("\n" + "=" * 70)
print("3/4 QUERY SQL (DistilGPT-2 + LoRA, entity-id context)")
print("=" * 70)
conn = build_sqlite_db(result["clean_df"])
sql_result = run_sql_generation_eval(train_by_task["query_sql"], val_by_task["query_sql"], conn)

print("\n" + "=" * 70)
print("4/4 SPEND SUMMARIZATION (DistilGPT-2 + LoRA, facts-in-prompt)")
print("=" * 70)
summ_result = run_summarization_eval(train_by_task["spend_summarization"], val_by_task["spend_summarization"], epochs=8)

print("\n\n" + "=" * 70)
print("FINAL SUMMARY - real CSV (diverse_transactions.csv), all fixes applied")
print("=" * 70)
print(f"{'Task':<22} {'Metric':<28} {'Value':>8}")
print(f"{'categorization':<22} {'val_accuracy':<28} {cat_result['accuracy']:>8.1%}")
print(f"{'fraud_flagging':<22} {'val_accuracy':<28} {fraud_result['accuracy']:>8.1%}")
print(f"{'fraud_flagging':<22} {'FRAUD recall':<28} {fraud_result['per_class_accuracy'].get('FRAUD', 0):>8.1%}")
print(f"{'query_sql':<22} {'executable_rate':<28} {sql_result['executable_rate']:>8.1%}")
print(f"{'query_sql':<22} {'correct_rate':<28} {sql_result['correct_rate']:>8.1%}")
print(f"{'spend_summarization':<22} {'fact_inclusion_rate':<28} {summ_result['avg_fact_inclusion_rate']:>8.1%}")

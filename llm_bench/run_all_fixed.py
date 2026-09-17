"""
Full benchmark with ALL fixes applied, in one run - for a clean screenshot.

Uses this same directory's eval_harness (entity-id context + wider token
budget for SQL, facts-in-prompt for summarization, oversampling+class-
weighting default for classification, context-order fix for fraud).

Needs torch/transformers/peft/datasets installed - this directory's own
venv does not have them; run with the Transformer_AI project's venv, e.g.:
    caffeinate -i /Users/ranayilmaz/Desktop/Transformer_AI/.venv/bin/python3 llm_bench/run_all_fixed.py
"""
import sys
sys.path.insert(0, "/Users/ranayilmaz/Desktop/txn_pipeline")
sys.path.insert(0, "/Users/ranayilmaz/Desktop/txn_pipeline/eval_harness")

from data_synth import generate_provider_a, generate_provider_b
from pipeline.orchestrator import run_pipeline
from pipeline.query_engine import build_sqlite_db
from external_llm_eval import run_classification_eval, run_sql_generation_eval, run_summarization_eval

ALL_TASKS = ["categorization", "fraud_flagging", "query_sql", "spend_summarization"]

train_by_task, val_by_task, clean_dfs = {t: [] for t in ALL_TASKS}, {t: [] for t in ALL_TASKS}, []
for gen in (generate_provider_a, generate_provider_b):
    raw = gen()
    result = run_pipeline(raw, gen.__name__, ALL_TASKS)
    for t in ALL_TASKS:
        examples = result["shaped_outputs"][t]
        train_by_task[t] += [e for e in examples if e["split"] == "train"]
        val_by_task[t] += [e for e in examples if e["split"] in ("val", "test")]
    clean_dfs.append(result["clean_df"])

print("=" * 70)
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
import pandas as pd
conn = build_sqlite_db(pd.concat(clean_dfs, ignore_index=True))
sql_result = run_sql_generation_eval(train_by_task["query_sql"], val_by_task["query_sql"], conn)  # default epochs=20, balanced

print("\n" + "=" * 70)
print("4/4 SPEND SUMMARIZATION (DistilGPT-2 + LoRA, facts-in-prompt)")
print("=" * 70)
summ_result = run_summarization_eval(train_by_task["spend_summarization"], val_by_task["spend_summarization"], epochs=8)

print("\n\n" + "=" * 70)
print("FINAL SUMMARY - all fixes applied")
print("=" * 70)
print(f"{'Task':<22} {'Metric':<28} {'Value':>8}")
print(f"{'categorization':<22} {'val_accuracy':<28} {cat_result['accuracy']:>8.1%}")
print(f"{'fraud_flagging':<22} {'val_accuracy':<28} {fraud_result['accuracy']:>8.1%}")
print(f"{'fraud_flagging':<22} {'FRAUD recall':<28} {fraud_result['per_class_accuracy'].get('FRAUD', 0):>8.1%}")
print(f"{'query_sql':<22} {'executable_rate':<28} {sql_result['executable_rate']:>8.1%}")
print(f"{'query_sql':<22} {'correct_rate':<28} {sql_result['correct_rate']:>8.1%}")
print(f"{'spend_summarization':<22} {'fact_inclusion_rate':<28} {summ_result['avg_fact_inclusion_rate']:>8.1%}")

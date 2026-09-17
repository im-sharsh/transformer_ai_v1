"""
Streamlit demo UI for the transaction preprocessing pipeline.
Walks through Stage 0 -> Stage A -> Stage B, showing what the pipeline
did and why, then gives downloadable clean output examples.
"""
import json
import io
import pandas as pd
import streamlit as st

from data_synth import generate_provider_a, generate_provider_b
from pipeline import run_pipeline, check_missing_fields, AUTO_INDEX_ENTITY_ID
from pipeline.schema import CANONICAL_FIELDS
from pipeline.tiny_transformer import run_training_sanity_check

CANONICAL_FIELDS_LIST = list(CANONICAL_FIELDS.keys())

st.set_page_config(page_title="Transaction Preprocessing Pipeline", layout="wide")

TASK_LABELS = {
    "fraud_flagging": "Fraud Flagging",
    "categorization": "Categorization",
    "spend_summarization": "Spend Summarization",
    "general_behavioral": "General Behavioral",
    "query_sql": "Natural-Language Query (NL→SQL)",
}

st.title("🏦 Universal Transaction Preprocessing Pipeline")
st.caption("Raw CSV → Schema Resolution → Cleaning → Task-Aware Shaping → LLM-ready output")

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Configuration")

source_choice = st.sidebar.radio(
    "Data source",
    ["Provider A (standard schema)", "Provider B (messy schema)", "Both (combined run)",
     "Upload my own CSV(s)"],
    help="Two synthetic providers with DIFFERENT raw schemas, to demonstrate that "
         "schema resolution works on arbitrary column names/conventions. "
         "Or upload your own CSV(s) to run the pipeline on real data.",
)

uploaded_files = None
if source_choice == "Upload my own CSV(s)":
    uploaded_files = st.sidebar.file_uploader(
        "Upload transaction CSV(s)",
        type=["csv"],
        accept_multiple_files=True,
        help="Each file is treated as a separate source and run through the "
             "pipeline independently, so files with different schemas are fine.",
    )

task_choice = st.sidebar.multiselect(
    "Task type(s) to shape output for",
    list(TASK_LABELS.keys()),
    default=list(TASK_LABELS.keys()),
    format_func=lambda k: TASK_LABELS[k],
)

st.sidebar.markdown("**Time-based train/val/test split**")

TASKS_NEEDING_TEST_SPLIT = {"fraud_flagging", "categorization", "query_sql"}
default_include_test = any(t in TASKS_NEEDING_TEST_SPLIT for t in task_choice)
include_test_split = st.sidebar.checkbox(
    "Include a held-out test split",
    value=default_include_test,
    help="Recommended for fraud flagging / categorization, which have clean "
         "metrics (precision/recall/accuracy) worth evaluating on held-out data. "
         "Less useful for summarization/general-behavioral, where evaluation is "
         "mostly qualitative — train/val is often enough there.",
)

train_pct = st.sidebar.slider("Train %", 50, 90, 70, step=5)
if include_test_split:
    val_pct = st.sidebar.slider("Val %", 5, 100 - train_pct - 5, min(15, 100 - train_pct - 5), step=5)
    test_pct = 100 - train_pct - val_pct
else:
    val_pct = 100 - train_pct
    test_pct = 0

st.sidebar.caption(
    f"Train {train_pct}% · Val {val_pct}%" + (f" · Test {test_pct}%" if test_pct else "")
    + " — split chronologically (earliest transactions train, most recent test/val), never randomly."
)

run_button = st.sidebar.button("▶ Run Pipeline", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data is synthetically generated for this demo. No real customer data is used."
)

if "overrides" not in st.session_state:
    st.session_state.overrides = {}
if "confirmed_sources" not in st.session_state:
    st.session_state.confirmed_sources = set()
if "extra_context_cols" not in st.session_state:
    st.session_state.extra_context_cols = {}

if run_button:
    st.session_state.pipeline_active = True
    st.session_state.overrides = {}
    st.session_state.confirmed_sources = set()

if not st.session_state.get("pipeline_active"):
    st.info("Configure options in the sidebar, then click **Run Pipeline**.")
    st.stop()

# ---------------------------------------------------------------- load data
sources = []
if source_choice.startswith("Provider A") or source_choice == "Both (combined run)":
    sources.append(("Provider A (standard)", generate_provider_a()))
if source_choice.startswith("Provider B") or source_choice == "Both (combined run)":
    sources.append(("Provider B (messy)", generate_provider_b()))

if source_choice == "Upload my own CSV(s)":
    if not uploaded_files:
        st.warning("Upload at least one CSV file in the sidebar, then click **Run Pipeline**.")
        st.stop()
    for f in uploaded_files:
        try:
            df = pd.read_csv(f)
        except Exception as strict_error:
            # A single malformed row (wrong field count, stray delimiter, etc.)
            # otherwise rejects the ENTIRE file even if every other row is
            # fine. Retry leniently, skipping only the rows that don't parse,
            # and tell the user exactly how many were dropped at this stage -
            # this is separate from and prior to Stage A's own row validation.
            try:
                f.seek(0)
                df = pd.read_csv(f, on_bad_lines="skip", engine="python")
                st.warning(
                    f"'{f.name}' had malformed row(s) that couldn't be parsed as CSV at all "
                    f"(wrong number of fields, stray delimiter, etc.) — those specific rows "
                    f"were skipped so the rest of the file could still be processed. "
                    f"Original error: {strict_error}"
                )
            except Exception as lenient_error:
                st.error(f"Could not read '{f.name}' as CSV even with lenient parsing: {lenient_error}")
                continue
        if df.empty:
            st.error(f"'{f.name}' has no rows — skipping.")
            continue
        sources.append((f.name, df))

if not sources:
    st.error("No valid data sources to run. Check the file(s) above and try again.")
    st.stop()

if not task_choice:
    st.warning("Select at least one task type in the sidebar.")
    st.stop()

for source_name, raw_df in sources:
    st.header(f"📄 Source: {source_name}")
    safe_name = "".join(c if c.isalnum() else "_" for c in source_name.split(".")[0]).strip("_")

    # -------- pre-check: can required + important-optional fields be auto-mapped? --------
    missing_required, missing_important_optional, _, candidate_cols = check_missing_fields(raw_df)
    already_confirmed = source_name in st.session_state.confirmed_sources
    fields_to_ask = sorted(missing_required) + sorted(missing_important_optional)

    if fields_to_ask and not already_confirmed:
        if missing_required:
            st.warning(
                f"⚠️ Could not automatically map required field(s) for **{source_name}**: "
                f"`{sorted(missing_required)}`. Pick the matching column(s) below to continue "
                f"— this avoids the pipeline guessing wrong or silently dropping this source."
            )
        if missing_important_optional:
            st.info(
                f"ℹ️ Could not automatically map: `{sorted(missing_important_optional)}`. These "
                f"aren't required — the pipeline can default them — but silently defaulting "
                f"`category` or `label` can quietly empty out the categorization/fraud-flagging "
                f"task output with no warning. Map them below if they exist in this file, or "
                f"leave as default if they genuinely don't."
            )
        st.dataframe(raw_df.head(3), use_container_width=True)

        with st.form(key=f"mapping_form_{source_name}"):
            selections = {}
            auto_index_confirmed = True  # default True; only matters if entity_id auto-index is chosen
            for field in fields_to_ask:
                is_required = field in missing_required
                skip_label = "-- not available in this file --" if is_required else "-- use default value --"
                options = [skip_label]
                if field == "entity_id":
                    options.append("🔢 No entity column — auto-index rows (1 row = 1 entity)")
                options += candidate_cols
                label_suffix = "" if is_required else " *(optional)*"
                selections[field] = st.selectbox(
                    f"Which column is **{field}**?{label_suffix}",
                    options=options,
                    key=f"select_{source_name}_{field}",
                )
                if field == "entity_id":
                    if selections[field].startswith("🔢 No entity column"):
                        st.warning(
                            "⚠️ **Before auto-indexing:** this should only be used when this "
                            "data genuinely has no account/customer concept — not just because "
                            "the column wasn't detected automatically. If these transactions "
                            "actually belong to real customers or accounts (even if the column "
                            "is unusually named, in a separate file, or needs to be joined in), "
                            "auto-indexing will silently discard those relationships, and every "
                            "transaction will be treated as if it happened in isolation. "
                            "Double-check the raw file for an account/customer identifier "
                            "before choosing this. Only confirm below if you're sure there is "
                            "no such identifier — not merely that this file doesn't show one."
                        )
                        auto_index_confirmed = st.checkbox(
                            "I've checked — there is genuinely no entity/account identifier "
                            "for this data, and I understand each transaction will be treated "
                            "independently.",
                            key=f"confirm_autoindex_{source_name}",
                        )
                    else:
                        st.caption(
                            "Auto-indexing (if chosen) treats every transaction as its own "
                            "independent entity — fine for categorization, but it removes the "
                            "entity grouping that fraud flagging, spend summarization, and "
                            "general behavioral sequencing rely on."
                        )
                elif field == "label":
                    st.caption("Needed for fraud flagging — without it, that task's output will be empty.")
                elif field == "category":
                    st.caption("Needed for categorization — without it, every transaction defaults to 'UNKNOWN'.")
            submitted = st.form_submit_button("Confirm mapping & continue")

        if submitted:
            skip_labels = {"-- not available in this file --", "-- use default value --"}
            wants_auto_index = selections.get("entity_id", "").startswith("🔢 No entity column")
            if wants_auto_index and not auto_index_confirmed:
                st.error("Please check the confirmation box above before proceeding with auto-indexing.")
                continue

            overrides = {}
            for f, c in selections.items():
                if c in skip_labels:
                    continue
                elif c.startswith("🔢 No entity column"):
                    overrides[f] = AUTO_INDEX_ENTITY_ID
                else:
                    overrides[f] = c
            st.session_state.overrides[source_name] = overrides
            st.session_state.confirmed_sources.add(source_name)
            st.rerun()
        else:
            continue  # don't run this source until the user confirms mapping

    manual_overrides = st.session_state.overrides.get(source_name, {})
    extra_cols_for_source = st.session_state.extra_context_cols.get(source_name, [])

    progress_bar = st.progress(0, text="Starting pipeline...")

    def _update_progress(label, frac):
        progress_bar.progress(min(max(frac, 0.0), 1.0), text=label)

    result = run_pipeline(
        raw_df, source_name, task_choice,
        manual_overrides=manual_overrides,
        train_frac=train_pct / 100, val_frac=val_pct / 100,
        progress_callback=_update_progress,
        extra_context_cols=extra_cols_for_source,
    )
    progress_bar.empty()

    if result["fatal_error"]:
        st.error(f"Pipeline halted for this source: {result['fatal_error']}")
        continue

    # ---------------- Dashboard summary (headline view before drilling into tabs)
    mapping_df_summary = pd.DataFrame(result["mapping_report"])
    status_counts = mapping_df_summary["status"].value_counts().to_dict()
    confident_ct = status_counts.get("auto_mapped", 0) + status_counts.get("manual_override", 0) + status_counts.get("auto_generated", 0)
    flagged_ct = status_counts.get("auto_mapped_flagged", 0)
    quarantined_ct = status_counts.get("quarantined", 0)

    stats = result["cleaning_stats"]
    retained_pct = (stats["output_rows"] / stats["input_rows"] * 100) if stats["input_rows"] else 0

    with st.container(border=True):
        st.markdown(f"### 📊 Summary — {source_name}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows in → out", f"{stats['input_rows']} → {stats['output_rows']}",
                   f"{retained_pct:.0f}% retained")
        m2.metric("Fields mapped confidently", f"{confident_ct}/9")
        m3.metric("Fields flagged / quarantined", f"{flagged_ct} / {quarantined_ct}")
        total_examples = sum(len(v) for v in result["shaped_outputs"].values())
        m4.metric("Total training examples", total_examples)

        task_cols = st.columns(len(task_choice)) if task_choice else []
        for col, t in zip(task_cols, task_choice):
            col.metric(TASK_LABELS[t], len(result["shaped_outputs"][t]))

        tiny_result_check = st.session_state.get(f"tiny_result_{source_name}")
        if tiny_result_check:
            verdict_icon = {"good": "✅", "weak": "⚠️", "none": "❌"}.get(tiny_result_check["verdict_level"], "")
            st.caption(f"{verdict_icon} Training sanity check: {tiny_result_check['verdict']}")
        else:
            st.caption("💡 Run the training sanity check in **Stage C** to see if this data has learnable signal.")

        if quarantined_ct > 0:
            st.caption(f"⚠️ {quarantined_ct} column(s) quarantined — see **Stage 0** for details and the "
                       f"option to carry any through as extra context.")

    tab0, tabA, tabB, tabC, tabM = st.tabs([
        "Stage 0 · Schema Resolution", "Stage A · Cleaning",
        "Stage B · Task Shaping", "Stage C · Training Sanity Check", "Manifest",
    ])

    # ---------------- Stage 0
    with tab0:
        st.subheader("Raw input preview")
        st.dataframe(raw_df.head(5), use_container_width=True)

        st.subheader("Column mapping")
        mapping_df = pd.DataFrame(result["mapping_report"])

        def _status_color(status):
            return {
                "auto_mapped": "🟢",
                "auto_mapped_flagged": "🟡",
                "used_for_sign_resolution": "🔵",
                "manual_override": "🟣",
                "auto_generated": "🔢",
                "quarantined": "🔴",
                "quarantined_collision": "🟠",
            }.get(status, "")

        mapping_df["status_icon"] = mapping_df["status"].apply(_status_color)
        display_cols = ["status_icon", "source_column", "canonical_field", "confidence", "method", "status"]
        if "note" in mapping_df.columns:
            display_cols.append("note")
        st.dataframe(
            mapping_df[display_cols],
            use_container_width=True,
        )
        st.caption("🟢 high-confidence auto-map · 🟡 auto-mapped but flagged for review · "
                   "🔵 used to resolve sign convention · 🟣 manually mapped by user · "
                   "🔢 auto-generated row index (no entity column available) · "
                   "🟠 lost a mapping tie to another column (see note) · "
                   "🔴 quarantined (not mapped)")

        # -------- plain-language confidence summary --------
        confident_fields = mapping_df[mapping_df["status"].isin(["auto_mapped", "manual_override"])]["canonical_field"].dropna().tolist()
        flagged_rows = mapping_df[mapping_df["status"] == "auto_mapped_flagged"][["canonical_field", "source_column", "confidence"]]
        summary_parts = []
        if confident_fields:
            summary_parts.append(f"**Confident about:** {', '.join(sorted(set(confident_fields)))}")
        if not flagged_rows.empty:
            flagged_desc = ", ".join(
                f"{r.canonical_field} (from `{r.source_column}`, {r.confidence:.0%} confidence)"
                for r in flagged_rows.itertuples()
            )
            summary_parts.append(f"**Worth double-checking:** {flagged_desc}")
        if summary_parts:
            st.markdown("  \n".join(summary_parts))

        # -------- manual override for any field, even already auto-mapped ones --------
        with st.expander("🔧 Override a field mapping (including auto-mapped ones)"):
            st.caption(
                "If a field mapped to the wrong column, fix it here. This works even for "
                "fields already auto-mapped with high confidence — automation can still be wrong."
            )
            override_field = st.selectbox(
                "Field to override", options=sorted(CANONICAL_FIELDS_LIST),
                key=f"override_field_{source_name}",
            )
            override_col = st.selectbox(
                "Use this raw column instead", options=["-- cancel --"] + list(raw_df.columns),
                key=f"override_col_{source_name}",
            )
            if st.button("Apply this override", key=f"apply_override_{source_name}"):
                if override_col != "-- cancel --":
                    current = dict(st.session_state.overrides.get(source_name, {}))
                    current[override_field] = override_col
                    st.session_state.overrides[source_name] = current
                    st.session_state.confirmed_sources.add(source_name)
                    st.rerun()

        if result["quarantined_columns"]:
            st.warning(f"Quarantined columns (left out of canonical schema): {result['quarantined_columns']}")

            st.markdown("**Carry any of these through as extra context?**")
            st.caption(
                "Quarantined only means 'doesn't fit the fixed schema fields' — it does NOT "
                "mean the column has no value for training. A column like a risk flag or "
                "transaction type could matter a lot, especially for fraud flagging. "
                "⚠️ Selected columns bypass this pipeline's PII-hashing and validation "
                "entirely — only pick columns you've already reviewed and are comfortable "
                "including as-is."
            )
            selected_extra = st.multiselect(
                "Include as extra context in shaped output",
                options=result["quarantined_columns"],
                default=extra_cols_for_source,
                key=f"extra_select_{source_name}",
            )
            if st.button("Apply & re-run with these extra columns", key=f"apply_extra_{source_name}"):
                st.session_state.extra_context_cols[source_name] = selected_extra
                st.rerun()
        else:
            st.success("No columns quarantined — all source columns resolved.")

        conventions = result["manifest"].data["schema_resolution"].get("convention_assumptions", [])
        if conventions:
            st.subheader("Detected conventions / assumptions applied")
            for c in conventions:
                st.write(f"**{c['field']}** — {c['assumption']}  \n*Confidence: {c['confidence']}*")

    # ---------------- Stage A
    with tabA:
        stats = result["cleaning_stats"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Input rows", stats["input_rows"])
        c2.metric("Dropped (missing required)", stats["dropped_missing_required"])
        c3.metric("Duplicates removed", stats["duplicate_rows_removed"])
        c4.metric("Output rows", stats["output_rows"])

        st.info(f"PII fields hashed (one-way, salted): {', '.join(stats['pii_fields_hashed'])}")

        if stats["dropped_missing_required"] > 0:
            st.error(
                f"⚠️ {stats['dropped_missing_required']} of {stats['input_rows']} rows were "
                f"dropped due to a missing required field after parsing. Breakdown by field "
                f"(null count after parsing, before dropping):"
            )
            null_df = pd.DataFrame(
                list(stats["null_counts_by_required_field"].items()),
                columns=["field", "null_count_after_parsing"],
            )
            st.dataframe(null_df, use_container_width=True)
            st.caption(
                "If a field shows a high null count here despite being mapped in Stage 0, "
                "the raw values likely have a formatting issue (unrecognized date format, "
                "non-numeric characters in amounts, etc.) rather than being genuinely missing. "
                "Check the raw preview in Stage 0 for that column's actual values."
            )

        st.subheader("Clean canonical data (with causal features)")
        display_cols = ["entity_id", "timestamp", "amount", "currency", "category",
                         "channel", "counterparty", "status", "day_of_week",
                         "hour_of_day", "time_since_last_txn_hours", "rolling_avg_amount_3"]
        extra_cols_present = [c for c in result["clean_df"].columns if c.startswith("extra__")]
        st.dataframe(result["clean_df"][display_cols + extra_cols_present].head(15), use_container_width=True)
        st.caption("`entity_id` and `counterparty` are shown post-hashing — this is the same "
                   "representation the model will see.")
        if extra_cols_present:
            st.info(f"Extra context columns included (unhashed, opt-in): "
                    f"{[c.replace('extra__','') for c in extra_cols_present]}")

        if result["split_stats"]:
            st.subheader("Time-based train / val / test split")
            sp = result["split_stats"]
            has_test = sp["row_counts"]["test"] > 0
            cols = st.columns(3 if has_test else 2)
            cols[0].metric("Train rows", sp["row_counts"]["train"])
            cols[1].metric("Val rows", sp["row_counts"]["val"])
            if has_test:
                cols[2].metric("Test rows", sp["row_counts"]["test"])
            cutoff_text = f"Train ends at **{sp['train_cutoff']}**"
            if has_test:
                cutoff_text += f" · Val ends at **{sp['val_cutoff']}** · everything after is test"
            else:
                cutoff_text += " · everything after is val (no held-out test split for this run)"
            st.caption(
                cutoff_text + ". Split by time, not randomly — the model is always "
                "evaluated on data that comes after what it trained on."
            )

    # ---------------- Stage B
    with tabB:
        for task_type in task_choice:
            examples = result["shaped_outputs"][task_type]
            st.subheader(TASK_LABELS[task_type])
            st.caption(f"{len(examples)} examples generated")

            if task_type == "query_sql" and examples:
                st.info(
                    "Every SQL query below was **executed against your real cleaned data** and "
                    "validated before being kept — none are guessed. The model is trained to "
                    "produce the SQL (translation), not to compute the answer itself."
                )
            if task_type == "spend_summarization" and examples:
                st.info(
                    "**Fact-grounded**: every number in each summary was computed via real SQL "
                    "aggregation (see 'facts_used' in each example) — only the sentence phrasing "
                    "is templated. This replaced the earlier placeholder-based version."
                )

            if not examples:
                st.write("No examples generated for this task (e.g. no labeled rows, or not enough "
                          "transaction history per entity for this task's requirements).")
                continue

            n_preview = min(3, len(examples))
            if task_type == "query_sql":
                for i in range(n_preview):
                    e = examples[i]
                    with st.expander(f"\"{e['instruction']}\"  (split: {e.get('split','unassigned')})",
                                      expanded=(i == 0)):
                        st.code(e["output"], language="sql")
                        st.caption(f"Executed result: {e['reference_answer']}")
            elif task_type == "spend_summarization":
                for i in range(n_preview):
                    e = examples[i]
                    with st.expander(f"\"{e['instruction']}\"  (split: {e.get('split','unassigned')})",
                                      expanded=(i == 0)):
                        st.write(e["output"])
                        st.caption("Underlying computed facts:")
                        st.json(e["facts_used"])
            else:
                for i in range(n_preview):
                    with st.expander(f"Example {i + 1} (split: {examples[i].get('split', 'unassigned')})",
                                      expanded=(i == 0)):
                        st.json(examples[i])

            by_split = {"train": [], "val": [], "test": []}
            for e in examples:
                by_split.setdefault(e.get("split", "unassigned"), []).append(e)

            dl_cols = st.columns(3)
            for col, split_name in zip(dl_cols, ["train", "val", "test"]):
                split_examples = by_split.get(split_name, [])
                jsonl_bytes = "\n".join(json.dumps(e) for e in split_examples).encode("utf-8")
                col.download_button(
                    label=f"⬇ {split_name} ({len(split_examples)})",
                    data=jsonl_bytes,
                    file_name=f"{safe_name}_{task_type}_{split_name}.jsonl",
                    mime="application/jsonl",
                    key=f"dl_{source_name}_{task_type}_{split_name}",
                    disabled=(len(split_examples) == 0),
                    use_container_width=True,
                )

    # ---------------- Stage C (training sanity check)
    with tabC:
        st.subheader("Does this data actually have learnable signal?")
        st.caption(
            "Trains a small transformer **from scratch** (no pretrained weights, "
            "no internet access needed) directly on your categorization output, "
            "to check whether it can learn the category from a transaction's "
            "other fields. This is a data sanity check, not a production training "
            "run — a real fine-tune of a real LLM is a separate, larger effort."
        )

        cat_examples = result["shaped_outputs"].get("categorization", [])
        train_ex = [e for e in cat_examples if e.get("split") == "train"]
        val_ex = [e for e in cat_examples if e.get("split") in ("val", "test")]

        if "categorization" not in task_choice:
            st.info("Select **Categorization** as a task type in the sidebar to enable this check "
                     "— it's the task with a clean, measurable signal (predict category from the rest "
                     "of the transaction).")
        elif len(train_ex) < 20 or len(val_ex) < 5:
            st.warning(f"Not enough categorization examples yet to train on "
                       f"(train={len(train_ex)}, val={len(val_ex)}). Need at least ~20 train / 5 val.")
        else:
            run_check = st.button("▶ Train tiny transformer on this data", key=f"train_check_{source_name}")
            if run_check:
                with st.spinner("Training a small transformer from scratch on your data..."):
                    tiny_result = run_training_sanity_check(train_ex, val_ex, epochs=25, lr=0.08)
                st.session_state[f"tiny_result_{source_name}"] = tiny_result

            tiny_result = st.session_state.get(f"tiny_result_{source_name}")
            if tiny_result:
                hist = tiny_result["history"]
                hist_df = pd.DataFrame(hist).set_index("epoch")

                verdict_fn = {"good": st.success, "weak": st.warning, "none": st.error}
                verdict_fn.get(tiny_result["verdict_level"], st.info)(tiny_result["verdict"])
                if tiny_result["small_sample_warning"] and tiny_result["verdict_level"] != "none":
                    st.caption(f"⚠️ {tiny_result['small_sample_warning']}")

                col1, col2, col3 = st.columns(3)
                col1.metric("Random-guess baseline", f"{tiny_result['random_baseline_accuracy']:.0%}")
                col2.metric("Majority-class baseline", f"{tiny_result['majority_baseline_accuracy']:.0%}")
                col3.metric("Final validation accuracy", f"{tiny_result['final_val_accuracy']:.0%}")
                st.caption(
                    "The majority-class baseline is the more honest bar to clear: it's what a "
                    "model that always guesses the most common category would score, without "
                    "learning anything. Beating random chance alone isn't enough."
                )

                st.line_chart(hist_df[["train_loss"]])
                st.caption("Training loss — should trend down if the data has learnable structure.")

                st.line_chart(hist_df[["train_accuracy", "val_accuracy"]])
                st.caption(
                    "Train vs. validation accuracy. If train accuracy keeps climbing while "
                    "validation accuracy stalls or drops, the model is memorizing training "
                    "examples rather than learning a pattern that generalizes (overfitting)."
                )

                st.markdown("**Per-class accuracy (validation set)**")
                pc_df = pd.DataFrame(
                    sorted(tiny_result["per_class_accuracy"].items(), key=lambda x: x[1]),
                    columns=["category", "accuracy"],
                )
                st.dataframe(pc_df, use_container_width=True, hide_index=True)
                st.caption(
                    "A model can post a good overall accuracy while quietly failing on rarer "
                    "categories — check for any category near 0% here even if the headline "
                    "number looks fine."
                )

                bc1, bc2 = st.columns(2)
                with bc1:
                    st.markdown("**Before training** (random weights)")
                    for s in tiny_result["before_samples"]:
                        correct = "✅" if s["predicted"] == s["true"] else "❌"
                        st.write(f"{correct} true=`{s['true']}` predicted=`{s['predicted']}` "
                                 f"(conf {s['confidence']})")
                with bc2:
                    st.markdown("**After training**")
                    for s in tiny_result["after_samples"]:
                        correct = "✅" if s["predicted"] == s["true"] else "❌"
                        st.write(f"{correct} true=`{s['true']}` predicted=`{s['predicted']}` "
                                 f"(conf {s['confidence']})")

                st.caption(
                    f"Trained on {tiny_result['train_examples_used']} examples, "
                    f"validated on {tiny_result['val_examples_used']} — subsampled for speed, "
                    f"not using the full dataset."
                )

    # ---------------- Manifest
    with tabM:
        st.subheader("Run manifest (full audit trail)")
        manifest_json = result["manifest"].to_json()
        st.json(result["manifest"].to_dict())
        st.download_button(
            label="⬇ Download manifest.json",
            data=manifest_json.encode("utf-8"),
            file_name=f"{safe_name}_manifest.json",
            mime="application/json",
            key=f"manifest_{source_name}",
        )

    st.markdown("---")

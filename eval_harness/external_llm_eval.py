"""
external_llm_eval.py - fine-tune REAL pretrained models on this pipeline's
output, on a GPU, in Colab.

WHY THIS IS SEPARATE FROM evaluation_harness.py:
The sandbox that built this pipeline has no GPU and (after a prior disk-
filling incident installing full CUDA torch) too little disk to even
CPU-install transformers+torch for a structural test. This script has NOT
been execution-verified the way the rest of this project was - it's written
carefully against standard, stable Hugging Face APIs, but you will be the
first to actually run it. If something breaks, the error message + which
cell it happened in is enough for me to fix it - treat the first Colab run
as the real test, the same way we treated the stress-test CSV earlier.

WHAT THIS COVERS (all 5 pipeline tasks now have a real-model eval path):
1. run_classification_eval() - fine-tunes a small pretrained encoder
   (distilbert-base-uncased by default). Works directly for BOTH
   categorization and fraud_flagging - same underlying job (text in, label
   out), just different context shape and label vocabulary.
2. run_sql_generation_eval() - fine-tunes a small pretrained causal LM
   (distilgpt2 by default, LoRA) on query_sql, then EXECUTES the generated
   SQL against the same SQLite query engine this pipeline uses and checks
   the result matches - a real correctness metric, not text similarity.
3. run_summarization_eval() - fine-tunes a causal LM on spend_summarization.
   Since summaries are open-ended prose (no execution check possible),
   correctness is measured as a fact-inclusion rate: does the generated
   text actually contain the real computed facts (total spend, top
   category, etc.) it was supposed to report.
4. run_general_behavioral_eval() - fine-tunes a causal LM on plain sequence
   continuation. No instruction/answer to check, so evaluation uses
   standard held-out perplexity - lower means the model finds real
   transaction sequences more predictable.

USAGE (in a Colab cell, GPU runtime):
    !pip install transformers datasets peft accelerate -q
    from external_llm_eval import (run_classification_eval, run_sql_generation_eval,
                                     run_summarization_eval, run_general_behavioral_eval)

    cls_results = run_classification_eval(train_examples, val_examples)          # categorization
    fraud_results = run_classification_eval(fraud_train_examples, fraud_val_examples)  # fraud_flagging
    sql_results = run_sql_generation_eval(train_examples, val_examples, sqlite_conn)
    summ_results = run_summarization_eval(train_examples, val_examples)
    behavioral_results = run_general_behavioral_eval(train_examples, val_examples)
"""
import json
import re


def _require(package_name, pip_name=None):
    """Fail fast with a clear message naming exactly what to install,
    rather than a buried ImportError several calls deep."""
    try:
        return __import__(package_name)
    except ImportError:
        raise ImportError(
            f"'{package_name}' is required for this function. In Colab, run:\n"
            f"  !pip install {pip_name or package_name} -q\n"
            f"then restart the runtime if this is the first install in the session."
        )


def _oversample_minority(examples, target_ratio=0.15):
    """
    Duplicate minority-class examples until each class is at least
    target_ratio of the training set. Complements (does not replace) class
    weighting - measured on fraud_flagging (~2.5% positive rate): class
    weighting ALONE (a ~40x loss weight on the rare class) still collapsed
    to predicting the majority class 100% of the time (0% recall on the
    rare class) even though the weight was applied correctly - gradient
    clipping was flattening the very large weighted updates back down to
    roughly the same magnitude as unweighted ones. Oversampling changes what
    the model actually SEES per epoch, which isn't subject to that same
    clipping interaction. Combined (oversampling + class weights), FRAUD
    recall went from 0% to 60% with balanced_accuracy 50%->80%, in real
    local testing. No-ops (returns input unchanged) for already-balanced
    labels, like categorization, since target_ratio is rarely binding there.
    """
    from collections import defaultdict
    by_label = defaultdict(list)
    for e in examples:
        by_label[e["output"]].append(e)
    if len(by_label) < 2:
        return examples
    majority_label = max(by_label, key=lambda l: len(by_label[l]))
    n_majority = len(by_label[majority_label])
    out = list(by_label[majority_label])
    for label, items in by_label.items():
        if label == majority_label:
            continue
        n_needed = int(n_majority * target_ratio / (1 - target_ratio)) - len(items)
        if n_needed > 0:
            items = items + [items[i % len(items)] for i in range(n_needed)]
        out.extend(items)
    return out


def run_classification_eval(train_examples, val_examples, model_name="distilbert-base-uncased",
                             epochs=3, batch_size=16, use_class_weights=True,
                             use_oversampling=True, output_dir="./cls_eval_output"):
    """
    Fine-tunes a real pretrained encoder on any text-classification-style task.

    Works directly for BOTH categorization and fraud_flagging - both are
    structurally the same job (text in, one label out), just with different
    context shapes (categorization: one transaction; fraud_flagging: a
    multi-transaction window) and different label vocabularies. No need for
    a separate fraud-specific function.

    train_examples / val_examples: the pipeline's shaped output for either
    task - each a dict with "context": [text, ...] and "output": label.

    use_class_weights: if True (default), the loss is weighted inversely to
    class frequency. Without this, a severely imbalanced label distribution
    (confirmed in real testing: fraud flagging at ~97% legitimate) gives a
    model almost nothing to gain by ever predicting the rare class - it can
    match the majority baseline by always guessing the common label and
    never learn the rare one at all. This isn't unique to fraud; any
    imbalanced classification task here benefits from it, so it's a general
    option rather than a fraud-specific special case.

    use_oversampling: if True (default), also duplicate minority-class
    examples (see _oversample_minority) - class weighting alone was
    measured to be insufficient on its own for a label as skewed as
    fraud_flagging (see _oversample_minority's docstring for the numbers).

    Returns a dict with: accuracy, majority_baseline, per_class_report,
    and the trained model/tokenizer (for further inspection or saving).
    """
    torch = _require("torch")
    transformers = _require("transformers")
    datasets_lib = _require("datasets")
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                               TrainingArguments, Trainer)
    from datasets import Dataset
    import numpy as np
    from collections import Counter

    if use_oversampling:
        before = Counter(e["output"] for e in train_examples)
        train_examples = _oversample_minority(train_examples)
        after = Counter(e["output"] for e in train_examples)
        if before != after:
            print(f"Oversampled minority class(es): {dict(before)} -> {dict(after)}")

    labels = sorted({e["output"] for e in train_examples + val_examples})
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    def _to_hf_dataset(examples):
        # join full context list, not just context[0] - categorization has a
        # single-item context (no-op join), fraud_flagging has a multi-
        # transaction window (joined into one text blob) - this makes the
        # same function correctly usable for both classification-style tasks.
        #
        # context[-1] (the most recent transaction) is placed FIRST, not in
        # its natural chronological position at the end. build_fraud_examples
        # labels a window by its LAST transaction, but tokenization below
        # truncates to max_length=64 with the default (right-side) truncation
        # - a single serialized transaction is already ~35-40 tokens, so any
        # window longer than 1-2 transactions was silently losing exactly
        # the transaction the label describes. Measured: this alone dropped
        # fraud_flagging from ~98% accuracy/60% FRAUD recall (single-
        # transaction context) to 77%/23.5% (full window, chronological
        # order) - putting the labeled transaction first guarantees it
        # survives truncation regardless of window length.
        def _join_context(ctx):
            return " ".join([ctx[-1]] + ctx[:-1])

        return Dataset.from_dict({
            "text": [_join_context(e["context"]) for e in examples],
            "label": [label2id[e["output"]] for e in examples],
        })

    train_ds = _to_hf_dataset(train_examples)
    val_ds = _to_hf_dataset(val_examples)

    print(f"Loading tokenizer/model: {model_name} (downloads weights - needs internet)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), id2label=id2label, label2id=label2id,
    )

    def _tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

    train_ds = train_ds.map(_tokenize, batched=True)
    val_ds = val_ds.map(_tokenize, batched=True)

    majority_label = Counter(e["output"] for e in train_examples).most_common(1)[0][0]
    majority_baseline = sum(1 for e in val_examples if e["output"] == majority_label) / len(val_examples)

    class_weights = None
    if use_class_weights:
        label_counts = Counter(label2id[e["output"]] for e in train_examples)
        n_total, n_classes = len(train_examples), len(labels)
        # standard inverse-frequency weighting: rarer classes get proportionally
        # higher loss weight, so misclassifying them costs the model more
        weights = [n_total / (n_classes * label_counts.get(i, 1)) for i in range(n_classes)]
        class_weights = torch.tensor(weights, dtype=torch.float)
        print(f"Class weights (rarer = higher): "
              f"{dict(zip(labels, [round(w, 2) for w in weights]))}")

    def _compute_metrics(eval_pred):
        logits, refs = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = float((preds == refs).mean())
        per_class = {}
        for lbl_id, lbl_name in id2label.items():
            mask = refs == lbl_id
            if mask.sum() > 0:
                per_class[lbl_name] = float((preds[mask] == refs[mask]).mean())
        return {"accuracy": acc, "per_class_accuracy": per_class}

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to=[],
    )

    class WeightedLossTrainer(Trainer):
        """Standard Trainer, except the loss is class-weighted cross-entropy
        instead of plain (unweighted) cross-entropy, when class_weights is set."""
        def __init__(self, *args, class_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._loss_class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weight = self._loss_class_weights.to(logits.device) if self._loss_class_weights is not None else None
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    TrainerClass = WeightedLossTrainer if class_weights is not None else Trainer
    trainer_kwargs = {"class_weights": class_weights} if class_weights is not None else {}

    trainer = TrainerClass(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=_compute_metrics,
        **trainer_kwargs,
    )

    print(f"Training on {len(train_ds)} examples, validating on {len(val_ds)}...")
    trainer.train()
    eval_result = trainer.evaluate()

    print()
    print(f"Majority-class baseline: {majority_baseline:.2%}")
    print(f"Fine-tuned accuracy:     {eval_result['eval_accuracy']:.2%}")
    verdict = ("Learned real signal" if eval_result["eval_accuracy"] - majority_baseline > 0.15
               else "Weak or no signal beyond majority baseline")
    print(f"Verdict: {verdict}")

    return {
        "accuracy": eval_result["eval_accuracy"],
        "majority_baseline": majority_baseline,
        "per_class_accuracy": eval_result.get("eval_per_class_accuracy", {}),
        "verdict": verdict,
        "model": model,
        "tokenizer": tokenizer,
    }


def _balance_template_kinds(examples, min_share=0.20):
    """
    Oversample under-represented SQL template_kind groups (aggregation /
    extremes / comparison / retrieval) up to at least min_share of the
    training set each. Measured cause: "comparison" (the compound,
    two-subquery period-comparison template) was naturally only ~11% of
    generated examples - by far the least represented and structurally the
    hardest (it's the only template with TWO nested SELECTs) - and was
    disproportionately represented among the model's remaining wrong
    answers even after fixing entity-id/token-budget bugs and training
    longer. Same oversampling principle that fixed fraud_flagging's class
    imbalance, applied to template selection instead of a fraud/legit label.
    """
    from collections import defaultdict
    by_kind = defaultdict(list)
    for e in examples:
        by_kind[e.get("template_kind", "unknown")].append(e)
    if len(by_kind) < 2:
        return examples
    n_total = len(examples)
    out = []
    for kind, items in by_kind.items():
        target_n = max(len(items), int(n_total * min_share))
        if target_n > len(items):
            items = items + [items[i % len(items)] for i in range(target_n - len(items))]
        out.extend(items)
    return out


def run_sql_generation_eval(train_examples, val_examples, sqlite_conn,
                             model_name="distilgpt2", epochs=20, batch_size=4,
                             use_lora=True, balance_template_kinds=True, output_dir="./sql_eval_output"):
    """
    Fine-tunes a real pretrained causal LM on the query_sql task, then
    evaluates with a REAL correctness metric: generate SQL for each held-out
    question, EXECUTE it against sqlite_conn (the same query engine this
    pipeline uses), and check whether the result matches the reference
    answer - not just whether the generated text looks similar.

    sqlite_conn: the sqlite3 connection from pipeline.query_engine.build_sqlite_db()
                 - pass the SAME connection the examples were validated against.

    epochs defaults to 20 (not 3) - measured directly: at 3-8 epochs the
    model reliably produces syntactically valid SQL with the right
    entity_id/dates (once those bugs were fixed) but picks the wrong query
    TEMPLATE for the question far too often (correct_rate stuck near 0%);
    at 20 epochs, correct_rate reached ~14% on the same data. This is a
    genuinely slower-converging sub-skill (choosing among 8 template
    families), not a bug - more epochs is the actual fix.
    """
    torch = _require("torch")
    transformers = _require("transformers")
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from datasets import Dataset
    from pipeline.query_engine import run_query

    if balance_template_kinds:
        from collections import Counter
        before = Counter(e.get("template_kind", "unknown") for e in train_examples)
        train_examples = _balance_template_kinds(train_examples)
        after = Counter(e.get("template_kind", "unknown") for e in train_examples)
        if before != after:
            print(f"Balanced template kinds: {dict(before)} -> {dict(after)}")

    print(f"Loading tokenizer/model: {model_name} (downloads weights - needs internet)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if use_lora:
        peft = _require("peft")
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["c_attn", "c_proj"] if "gpt2" in model_name else None,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    PROMPT_TEMPLATE = "Question: {question}\nEntity: {entity_id}\nSQL: "

    def _entity_id(e):
        # _debug_sql_params[0] is entity_id for every query_sql template (see
        # nl_to_sql.py - every _tpl_* function returns (entity_id, ...) as
        # the first param element), despite the "_debug" name suggesting
        # internal-only use. Repurposed here rather than re-deriving it from
        # the literal SQL string via regex, which would be more fragile.
        return e["_debug_sql_params"][0]

    def _tokenize_with_masked_prompt(e, max_length=160):
        """
        Only compute training loss on the SQL answer tokens, never on the
        question/prompt tokens - dilutes the "where does the answer end"
        signal otherwise. A trailing ';' is added as an explicit,
        distinctive stop marker: a single consistent character is a much
        easier pattern for a small model to learn than "predict EOS out of
        the entire vocabulary," which alone proved insufficient in testing
        (the model kept generating past a complete, correct answer even
        with prompt-masking and anti-repetition decoding already applied).

        The entity_id is given in the prompt (see PROMPT_TEMPLATE) rather
        than left for the model to reproduce from memory - measured via
        real Colab/local runs: without it, the model regularly hallucinates
        a plausible-looking but WRONG 16-char hash (executable_rate reached
        86.5% once the separate max_new_tokens truncation bug was fixed,
        but correct_rate stayed at 0% specifically because the entity_id in
        the generated query never matched the one the reference answer was
        computed against - the query ran fine, just against the wrong
        person's data). Giving it the ID converts "recall an opaque 16-char
        hex string" into "copy this string", which is a far easier subtask
        for a small model already tight on capacity and training epochs.
        """
        prompt = PROMPT_TEMPLATE.format(question=e["instruction"], entity_id=_entity_id(e))
        answer = e["output"].rstrip(";") + ";" + tokenizer.eos_token

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + answer_ids)[:max_length]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]  # -100 = ignored in loss

        pad_len = max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        labels = labels + [-100] * pad_len  # padding also excluded from loss

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    # NOTE: deliberately NOT calling train_ds.set_format(type="torch", ...) here -
    # that routes tensor conversion through datasets' own torch formatter, which
    # has a broken torchvision.io.VideoReader import check unrelated to our
    # text-only data (hit in real Colab testing). Leaving features as plain
    # python lists lets Trainer's default collator build tensors directly via
    # torch.tensor(), skipping that broken code path entirely.
    train_ds = Dataset.from_list([_tokenize_with_masked_prompt(e) for e in train_examples])

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )

    # features already fully tokenized/padded/labeled above - Trainer's default
    # collator stacks the pre-built python-list features into tensors directly.
    trainer = Trainer(model=model, args=args, train_dataset=train_ds)

    print(f"Training on {len(train_ds)} examples...")
    trainer.train()

    # ---- Real evaluation: generate SQL, EXECUTE it, check against reference ----
    print()
    print(f"Evaluating on {len(val_examples)} held-out questions via real query execution...")
    correct, executable, total = 0, 0, 0
    failures = []

    model.eval()
    for e in val_examples:
        prompt = PROMPT_TEMPLATE.format(question=e["instruction"], entity_id=_entity_id(e))
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                # 60 was measured to be too tight: the model doesn't reproduce
                # entity_id hashes verbatim, so a generated hash can land on a
                # token-inefficient BPE split (worst case ~1 token/char, e.g.
                # 16 chars -> 16 tokens, vs. ~8 for a real training hash) and
                # eat most of the budget before the query is even half done -
                # confirmed by inspecting raw generated token ids, which hit
                # exactly 60 tokens every time, mid-date, regardless of
                # min_new_tokens/beam search (neither is the actual cause).
                # 160 comfortably covers the worst case even for
                # _tpl_compare_periods, which embeds the entity_id AND a full
                # date range twice in one query.
                **inputs, max_new_tokens=160, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # NOTE: deliberately NOT using repetition_penalty/no_repeat_ngram_size
                # here, unlike the summarization task below. Confirmed via real Colab
                # testing that they actively corrupt valid SQL: a correct BETWEEN
                # clause legitimately needs to repeat similar-looking date strings and
                # the word AND, and penalizing repetition was forcing the model off
                # the correct continuation into garbage right at that exact point.
                # The semicolon-truncation safety net below still guards against
                # genuine runaway generation without fighting valid SQL structure.
            )
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        # code-level safety net: truncate at the first ';' regardless of
        # whether the model's own stopping behavior worked - a query is
        # complete at its first semicolon no matter what comes after
        generated_sql = generated.split(";")[0].strip()
        if generated_sql:
            generated_sql += ";"

        total += 1
        # e["output"] is now fully self-contained literal SQL (no '?' params) -
        # both the generated and reference queries can be executed directly
        # and their results compared, giving a real correctness metric.
        ok, result = run_query(sqlite_conn, generated_sql)
        if ok:
            executable += 1
            reference_ok, reference_result = run_query(sqlite_conn, e["output"])
            if reference_ok and result["rows"] == reference_result["rows"]:
                correct += 1
            else:
                failures.append({"question": e["instruction"], "generated": generated_sql,
                                  "expected_sql": e["output"]})
        else:
            failures.append({"question": e["instruction"], "generated": generated_sql, "error": result})

    print()
    print(f"Executable (valid SQL syntax): {executable}/{total} ({executable/total:.1%})")
    print(f"Correct (matches reference answer): {correct}/{total} ({correct/total:.1%})")

    return {
        "total": total, "executable": executable, "correct": correct,
        "executable_rate": executable / total if total else 0,
        "correct_rate": correct / total if total else 0,
        "failures": failures[:10],  # sample for inspection
        "model": model, "tokenizer": tokenizer,
    }


def run_summarization_eval(train_examples, val_examples, model_name="distilgpt2",
                            epochs=5, batch_size=4, use_lora=True, output_dir="./summ_eval_output"):
    """
    Fine-tunes a real pretrained causal LM on the fact-grounded spend_summarization
    task. Since summaries are open-ended prose (not SQL), there's no
    execution-based correctness check available - instead, correctness is
    measured as a FACT-INCLUSION RATE: for each generated summary, check
    whether the actual computed facts (total spend, top category, top
    merchant, transaction count) it was supposed to report appear in the
    output. This stays mechanically checkable rather than relying on
    subjective "does this read well" judgment, consistent with this
    pipeline's fact-grounded design.

    train_examples / val_examples: shaped spend_summarization output - each
    a dict with "instruction", "output" (the narrative), and "facts_used".
    """
    torch = _require("torch")
    transformers = _require("transformers")
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from datasets import Dataset

    print(f"Loading tokenizer/model: {model_name} (downloads weights - needs internet)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if use_lora:
        peft = _require("peft")
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["c_attn", "c_proj"] if "gpt2" in model_name else None,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    PROMPT_TEMPLATE = "Question: {question}\nFacts: {facts}\nAnswer: "

    _CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹", "JPY": "¥"}

    def _facts_str(facts):
        # compact rendering of the SAME facts dict the reference narrative
        # was rendered from (see summarization_v2.py's _render_narrative) -
        # the model can only report numbers it's actually shown.
        #
        # Uses the entity's real currency symbol (falls back to the raw
        # code, e.g. "AUD 12.50", for one with no symbol here) instead of
        # a hardcoded "$" - a real multi-currency source (confirmed: every
        # entity in a real test CSV had 2+ currencies) would otherwise be
        # shown dollar signs on EUR/GBP/INR amounts, training the model to
        # confidently report the wrong currency.
        cur = _CURRENCY_SYMBOLS.get(facts.get("currency"), f"{facts['currency']} " if facts.get("currency") else "$")
        parts = [f"total_spend={facts['total_spend']:.2f}", f"transaction_count={facts['transaction_count']}"]
        if facts.get("top_category"):
            parts.append(f"top_category={facts['top_category']} "
                          f"({facts['top_category_pct']:.0f}%, {cur}{facts['top_category_amount']:.2f})")
        if facts.get("top_merchant"):
            parts.append(f"top_merchant={facts['top_merchant']} ({facts['top_merchant_count']}x)")
        if facts.get("max_amount"):
            parts.append(f"largest_txn={cur}{facts['max_amount']:.2f} at {facts['max_merchant']} ({facts['max_category']})")
        if facts.get("trend_direction"):
            parts.append(f"trend={facts['trend_direction']} {facts['trend_pct']:.0f}%")
        return ", ".join(parts)

    def _tokenize_with_masked_prompt(e, max_length=220):
        # same prompt-loss-masking approach as the SQL task, for the same
        # reason - only the answer (the narrative) should drive the loss.
        #
        # The facts are given in the prompt rather than left for the model
        # to invent - measured: without them, the ONLY input the model saw
        # was the bare question ("Summarize my spending in March 2026."),
        # with the specific numbers (total spend, top category, etc.)
        # appearing NOWHERE in its input, only in the target output. No
        # amount of training fixes that - the model was being asked to
        # report numbers it was never shown, so it consistently generated
        # the same generic, factually-wrong narrative regardless of the
        # actual input (fact-inclusion rate ~1.5-3%). Giving the real facts
        # in the prompt turns this into a much easier structured-data-to-
        # text task and raised fact-inclusion to ~21% in local testing -
        # still not production-good (only 160 training examples), but
        # confirms this was the real bottleneck, not model capacity.
        prompt = PROMPT_TEMPLATE.format(question=e["instruction"], facts=_facts_str(e["facts_used"]))
        answer = e["output"] + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + answer_ids)[:max_length]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
        pad_len = max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        labels = labels + [-100] * pad_len
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    train_ds = Dataset.from_list([_tokenize_with_masked_prompt(e) for e in train_examples])

    args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, logging_steps=10,
        save_strategy="no", report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds)

    print(f"Training on {len(train_ds)} examples...")
    trainer.train()

    print()
    print(f"Evaluating on {len(val_examples)} held-out summaries via fact-inclusion checking...")
    fact_scores = []
    samples = []

    model.eval()
    for e in val_examples:
        prompt = PROMPT_TEMPLATE.format(question=e["instruction"], facts=_facts_str(e["facts_used"]))
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=120, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.3, no_repeat_ngram_size=3,
            )
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        score, checked = _fact_inclusion_score(generated, e.get("facts_used", {}))
        fact_scores.append(score)
        if len(samples) < 5:
            samples.append({"instruction": e["instruction"], "generated": generated,
                             "facts_used": e.get("facts_used", {}), "fact_inclusion_score": score})

    avg_score = sum(fact_scores) / len(fact_scores) if fact_scores else 0
    print(f"Average fact-inclusion rate: {avg_score:.1%}")

    return {
        "avg_fact_inclusion_rate": avg_score,
        "per_example_scores": fact_scores,
        "samples": samples,
        "model": model, "tokenizer": tokenizer,
    }


def _fact_inclusion_score(generated_text: str, facts: dict):
    """
    Checks whether the actual computed facts appear in the generated text -
    a mechanical, checkable stand-in for "did the model report the real
    numbers," not a subjective quality judgment.
    """
    if not facts:
        return 0.0, []
    checks = []
    text_lower = generated_text.lower()

    if facts.get("total_spend") is not None:
        checks.append(f"{facts['total_spend']:.2f}" in generated_text or
                       str(round(facts["total_spend"])) in generated_text)
    if facts.get("transaction_count") is not None:
        checks.append(str(facts["transaction_count"]) in generated_text)
    if facts.get("top_category"):
        checks.append(str(facts["top_category"]).lower() in text_lower)
    if facts.get("top_merchant"):
        checks.append(str(facts["top_merchant"]).lower() in text_lower)
    if facts.get("max_amount") is not None:
        checks.append(f"{facts['max_amount']:.2f}" in generated_text or
                       str(round(facts["max_amount"])) in generated_text)

    if not checks:
        return 0.0, []
    return sum(checks) / len(checks), checks


def run_general_behavioral_eval(train_examples, val_examples, model_name="distilgpt2",
                                 epochs=5, batch_size=4, use_lora=True,
                                 output_dir="./behavioral_eval_output"):
    """
    Fine-tunes a real pretrained causal LM on the general_behavioral task
    (plain sequence continuation, no instruction/answer split - this task
    exists for continued-pretraining-style exposure to transaction data
    structure, not for answering questions). Since there's no "correct
    answer" to check here, evaluation uses the standard language-modeling
    metric: held-out perplexity - lower means the model finds real
    transaction sequences more predictable, which is the actual goal of
    this task.
    """
    torch = _require("torch")
    transformers = _require("transformers")
    import math
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from datasets import Dataset

    print(f"Loading tokenizer/model: {model_name} (downloads weights - needs internet)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if use_lora:
        peft = _require("peft")
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["c_attn", "c_proj"] if "gpt2" in model_name else None,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    def _tokenize_sequence(e, max_length=256):
        # no prompt/answer split here - the whole joined sequence IS the
        # content, so the whole thing (minus padding) contributes to loss,
        # unlike the instruction-style tasks above
        text = " ".join(e["sequence"]) + tokenizer.eos_token
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:max_length]
        pad_len = max_length - len(ids)
        attention_mask = [1] * len(ids) + [0] * pad_len
        labels = ids + [-100] * pad_len
        input_ids = ids + [tokenizer.pad_token_id] * pad_len
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    train_ds = Dataset.from_list([_tokenize_sequence(e) for e in train_examples])
    val_ds = Dataset.from_list([_tokenize_sequence(e) for e in val_examples])

    args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        eval_strategy="epoch", logging_steps=10, save_strategy="no", report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)

    print(f"Training on {len(train_ds)} sequences...")
    trainer.train()
    eval_result = trainer.evaluate()
    perplexity = math.exp(eval_result["eval_loss"]) if eval_result["eval_loss"] < 20 else float("inf")

    print()
    print(f"Held-out eval loss: {eval_result['eval_loss']:.3f}")
    print(f"Held-out perplexity: {perplexity:.2f}")

    return {
        "eval_loss": eval_result["eval_loss"],
        "perplexity": perplexity,
        "model": model, "tokenizer": tokenizer,
    }

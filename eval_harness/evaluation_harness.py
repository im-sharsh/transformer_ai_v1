"""
Evaluation harness: run multiple datasets x multiple model configs x multiple
seeds, and produce one comparable results table - instead of scattered
one-off runs that are hard to compare apples-to-apples.

This module only depends on numpy + the pipeline itself - it runs anywhere
(this sandbox, Colab, your laptop), no GPU or internet required. It's the
"from-scratch transformer" side of the comparison; see external_llm_eval.py
for the real-pretrained-model side, which needs GPU + internet (Colab).
"""
import time
import pandas as pd
import numpy as np

from pipeline.tiny_transformer import run_training_sanity_check


def prepare_datasets(sources: dict, task_types=("categorization",), train_frac=0.7, val_frac=0.15):
    """
    sources: {dataset_name: raw_dataframe}
    Runs each source through the full pipeline ONCE and caches the shaped
    output - so adding a new model config later doesn't require re-running
    Stage 0-B for every dataset again.

    Returns: {dataset_name: pipeline_result_dict}
    """
    from pipeline import run_pipeline
    cached = {}
    for name, df in sources.items():
        result = run_pipeline(df, name, list(task_types), train_frac=train_frac, val_frac=val_frac)
        if result["fatal_error"]:
            print(f"WARNING: dataset '{name}' failed to process: {result['fatal_error']}")
            continue
        cached[name] = result
    return cached


# A few sensible model-size presets to compare, spanning tiny -> more capacity.
# All still small enough to train in seconds on CPU - this isn't about
# absolute scale, it's about seeing whether more capacity actually helps
# on YOUR data, which is often not obvious in advance.
MODEL_CONFIGS = {
    "tiny (d=16)": dict(d_model=16, max_len=48),
    "small (d=32)": dict(d_model=32, max_len=48),
    "medium (d=64)": dict(d_model=64, max_len=64),
}


def run_grid(cached_datasets: dict, model_configs: dict = None, task_type="categorization",
             seeds=(0, 1, 2), epochs=25, lr=0.08):
    """
    Runs every (dataset x model_config x seed) combination and returns a
    long-format results DataFrame - one row per run, ready for pivoting
    into a comparison table or heatmap.

    Multiple seeds per cell matter: with small datasets (likely here),
    differences between model configs can be noise, not signal. Look at
    the spread across seeds, not just one run's number, before concluding
    one config is genuinely better than another.
    """
    model_configs = model_configs or MODEL_CONFIGS
    rows = []

    for dataset_name, result in cached_datasets.items():
        examples = result["shaped_outputs"].get(task_type, [])
        train_ex = [e for e in examples if e.get("split") == "train"]
        val_ex = [e for e in examples if e.get("split") in ("val", "test")]

        if len(train_ex) < 20 or len(val_ex) < 5:
            print(f"Skipping '{dataset_name}': not enough examples "
                  f"(train={len(train_ex)}, val={len(val_ex)})")
            continue

        for config_name, config_kwargs in model_configs.items():
            for seed in seeds:
                start = time.time()
                out = run_training_sanity_check(
                    train_ex, val_ex, epochs=epochs, lr=lr, seed=seed, **config_kwargs,
                )
                elapsed = time.time() - start

                rows.append({
                    "dataset": dataset_name,
                    "model_config": config_name,
                    "seed": seed,
                    "train_examples": out["train_examples_used"],
                    "val_examples": out["val_examples_used"],
                    "random_baseline": out["random_baseline_accuracy"],
                    "majority_baseline": out["majority_baseline_accuracy"],
                    "final_val_accuracy": out["final_val_accuracy"],
                    "final_train_accuracy": out["final_train_accuracy"],
                    "margin_over_majority": (out["final_val_accuracy"] - out["majority_baseline_accuracy"])
                                             if out["final_val_accuracy"] is not None and out["majority_baseline_accuracy"] is not None else None,
                    "verdict": out["verdict_level"],
                    "train_time_sec": round(elapsed, 2),
                })

    return pd.DataFrame(rows)


def summarize_grid(results_df: pd.DataFrame):
    """
    Collapse seeds into mean +/- std per (dataset, model_config) cell -
    the actual comparable summary, since a single seed's number can mislead.
    """
    if results_df.empty:
        return results_df
    summary = results_df.groupby(["dataset", "model_config"]).agg(
        val_accuracy_mean=("final_val_accuracy", "mean"),
        val_accuracy_std=("final_val_accuracy", "std"),
        margin_over_majority_mean=("margin_over_majority", "mean"),
        train_time_mean_sec=("train_time_sec", "mean"),
        n_seeds=("seed", "count"),
    ).reset_index()
    summary["val_accuracy_std"] = summary["val_accuracy_std"].fillna(0.0)
    return summary.sort_values(["dataset", "val_accuracy_mean"], ascending=[True, False])


def plot_comparison_heatmap(summary_df: pd.DataFrame, value_col="val_accuracy_mean", save_path=None):
    """
    Dataset x model_config heatmap - the fastest way to see which
    combinations actually worked. Returns the matplotlib figure (renders
    inline automatically in Colab/Jupyter; pass save_path to also write a file).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    pivot = summary_df.pivot(index="dataset", columns="model_config", values=value_col)
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.8), max(3, len(pivot.index) * 1.2)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=max(0.5, np.nanmax(pivot.values)), aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=10)

    ax.set_title(f"{value_col} by dataset x model config")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

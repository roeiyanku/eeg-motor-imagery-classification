from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..io.config import CLASS_NAMES
from ..results import EvalResult

# Winning kappa of the 2008 competition (FBCSP, Ang et al.), for reference lines.
BENCHMARK_KAPPA_2008 = 0.57


def write_metrics(results: list[EvalResult], output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "model": result.model,
            "subject": result.subject,
            "accuracy": result.accuracy,
            "macro_f1": result.macro_f1,
        }
        for result in results
    ]
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "metrics.csv", index=False)
    return df


def plot_class_distribution(y: np.ndarray, output_dir: Path) -> None:
    counts = pd.Series(y).map(lambda item: CLASS_NAMES[int(item)]).value_counts().reindex(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_title("Class distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Trials")
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=160)
    plt.close(fig)


def plot_model_comparison(metrics: pd.DataFrame, output_dir: Path) -> None:
    summary = metrics.groupby("model", as_index=False)["accuracy"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=summary, x="model", y="accuracy", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Mean within-subject accuracy")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.close(fig)


def plot_confusion_matrices(results: list[EvalResult], output_dir: Path) -> None:
    for model_name in sorted({result.model for result in results}):
        total = sum((result.confusion for result in results if result.model == model_name), np.zeros((4, 4), dtype=int))
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(total, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_title(f"{model_name} confusion matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.tight_layout()
        fig.savefig(output_dir / f"confusion_{model_name}.png", dpi=160)
        plt.close(fig)


def write_summary(metrics: pd.DataFrame, output_dir: Path) -> None:
    summary = metrics.groupby("model")[["accuracy", "macro_f1"]].agg(["mean", "std"]).round(3)
    lines = [
        "# Experiment Summary",
        "",
        "Evaluation: within-subject stratified train/test split across labeled Dataset 2a training files.",
        "",
        "```text",
        summary.to_string(),
        "```",
        "",
        "The evaluation files (`*E.gdf`) were not scored because their annotations contain `783 = unknown_cue` without public labels in the local files.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_benchmark(results, output_dir: Path) -> pd.DataFrame:
    """Persist per-subject T->E benchmark metrics and a kappa summary vs 2008."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "model": r.model,
            "subject": r.subject,
            "accuracy": r.accuracy,
            "kappa": r.kappa,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "benchmark_metrics.csv", index=False)

    summary = (
        df.groupby("model")[["accuracy", "kappa"]]
        .agg(["mean", "std"])
        .round(3)
    )

    # Per-model kappa vs the 2008 winner.
    mean_kappa = df.groupby("model")["kappa"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    colors = ["#2a9d8f" if v >= BENCHMARK_KAPPA_2008 else "#e76f51" for v in mean_kappa.values]
    ax.bar(list(mean_kappa.index), list(mean_kappa.values), color=colors)
    ax.axhline(BENCHMARK_KAPPA_2008, color="black", linestyle="--", linewidth=1.2)
    ax.text(
        len(mean_kappa) - 0.5,
        BENCHMARK_KAPPA_2008 + 0.01,
        f"2008 winner (FBCSP) = {BENCHMARK_KAPPA_2008}",
        ha="right",
        va="bottom",
        fontsize=9,
    )
    ax.set_ylim(0, max(0.8, float(mean_kappa.max()) + 0.1))
    ax.set_title("Competition benchmark: mean kappa on evaluation set (train T → test E)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Cohen's kappa")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / "benchmark_kappa.png", dpi=160)
    plt.close(fig)

    lines = [
        "# Competition Benchmark (train on T, test on E)",
        "",
        "Protocol: for each subject, train on all labeled calibration trials "
        "(`A0XT.gdf`) and score on the held-out evaluation trials (`A0XE.gdf`), "
        "using the official evaluation labels in `true_labels/`. Kappa is averaged "
        "across the nine subjects, matching the metric reported for the competition.",
        "",
        f"**2008 winning kappa (FBCSP, Ang et al.): {BENCHMARK_KAPPA_2008}**",
        "",
        "```text",
        summary.to_string(),
        "```",
        "",
        "Per-model mean kappa:",
        "",
    ]
    for model, value in mean_kappa.items():
        verdict = "beats" if value >= BENCHMARK_KAPPA_2008 else "below"
        lines.append(f"- `{model}`: kappa = {value:.3f} ({verdict} 2008)")
    (output_dir / "benchmark_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return df


def write_cnn_history(results: list[EvalResult], output_dir: Path) -> None:
    rows = []
    for result in results:
        for row in result.history:
            rows.append({"model": result.model, "subject": result.subject, **row})
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "cnn_history.csv", index=False)

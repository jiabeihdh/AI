import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_EXPERIMENTS = {
    "funsearch": Path("logs/baseline_full/summary.json"),
    "moe_static": Path("logs/moe_static_full/summary.json"),
    "moe_static_dedup": Path("logs/moe_static_dedup_full/summary.json"),
    "moe_dynamic_dedup": Path("logs/moe_dynamic_dedup_full/summary.json"),
}


def load_summary(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_summary_frame(experiments: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, path in experiments.items():
        payload = load_summary(path)
        search = payload.get("search_diagnostics", {})
        rows.append(
            {
                "experiment": label,
                "average_single_best_makespan": payload.get("average_single_best_makespan"),
                "average_single_best_relative_improvement": payload.get("average_single_best_relative_improvement"),
                "robust_single_best_score": payload.get("robust_single_best_score"),
                "average_fusion_makespan": payload.get("average_fusion_makespan"),
                "average_fusion_relative_improvement": payload.get("average_fusion_relative_improvement"),
                "average_fusion_gain_vs_single_best": payload.get("average_fusion_gain_vs_single_best"),
                "robust_fusion_score": payload.get("robust_fusion_score"),
                "fusion_wins": payload.get("fusion_wins"),
                "fusion_ties": payload.get("fusion_ties"),
                "fusion_losses": payload.get("fusion_losses"),
                "step_level_insertion_agreement_rate": payload.get("step_level_insertion_agreement_rate"),
                "kept_experts": search.get("kept_experts"),
                "merged_raw_candidate_count": search.get("merged_raw_candidate_count"),
                "merged_unique_candidate_count": search.get("merged_unique_candidate_count"),
                "merged_exact_dedup_ratio": search.get("merged_exact_dedup_ratio"),
                "merged_candidate_compression": search.get("merged_candidate_compression"),
                "merged_unique_compression": search.get("merged_unique_compression"),
                "merged_raw_to_unique_redundancy_ratio": search.get("merged_raw_to_unique_redundancy_ratio"),
                "merged_committee_unique_fingerprints": search.get("merged_committee_unique_fingerprints"),
                "merged_committee_unique_utilization_ratio": search.get("merged_committee_unique_utilization_ratio"),
                "merged_fingerprinting_elapsed_seconds": search.get("merged_fingerprinting_elapsed_seconds"),
                "merged_dedup_elapsed_seconds": search.get("merged_dedup_elapsed_seconds"),
                "mean_committee_diversity": search.get("mean_committee_diversity"),
                "dedup_mode": payload.get("dedup_mode", search.get("dedup_mode")),
                "fusion_mode": payload.get("fusion_mode", search.get("fusion_mode")),
            }
        )
    return pd.DataFrame(rows)


def prepare_display_metrics(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["plot_makespan"] = enriched["average_fusion_makespan"].where(
        enriched["average_fusion_makespan"].notna(),
        enriched["average_single_best_makespan"],
    )
    enriched["plot_robust_score"] = enriched["robust_fusion_score"].where(
        enriched["robust_fusion_score"].notna(),
        enriched["robust_single_best_score"],
    )
    return enriched


def save_tables(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = prepare_display_metrics(df)

    performance_df = pd.DataFrame(
        {
            "experiment": df["experiment"],
            "avg_makespan": df["plot_makespan"],
            "robust_score_pct": df["plot_robust_score"] * 100.0,
            "fusion_gain_vs_single_best": df["average_fusion_gain_vs_single_best"],
        }
    )

    efficiency_df = df[df["average_fusion_makespan"].notna()].copy()
    efficiency_df = efficiency_df[
        [
            "experiment",
            "merged_unique_candidate_count",
            "merged_exact_dedup_ratio",
            "merged_committee_unique_utilization_ratio",
            "mean_committee_diversity",
        ]
    ]

    performance_df.to_csv(output_dir / "performance_table.csv", index=False)
    performance_df.to_markdown(output_dir / "performance_table.md", index=False)
    efficiency_df.to_csv(output_dir / "efficiency_table.csv", index=False)
    efficiency_df.to_markdown(output_dir / "efficiency_table.md", index=False)


def plot_bar(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    percent: bool = False,
) -> None:
    values = df[y].astype(float)
    if percent:
        values = values * 100.0
    ax.bar(df[x], values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)


def save_plots(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = prepare_display_metrics(df)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    plot_bar(
        axes[0],
        df,
        "experiment",
        "plot_makespan",
        "Average makespan across experiments",
        "Makespan",
        percent=False,
    )
    plot_bar(axes[1], df, "experiment", "plot_robust_score", "Robust score", "Score (%)", percent=True)
    fig.tight_layout()
    fig.savefig(output_dir / "performance_main.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    moe_df = df[df["average_fusion_gain_vs_single_best"].notna()].copy()
    if not moe_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        plot_bar(axes[0], moe_df, "experiment", "average_fusion_gain_vs_single_best", "Fusion gain vs single-best", "Makespan gain")
        plot_bar(axes[1], moe_df, "experiment", "robust_fusion_score", "Robust fusion score", "Score")
        fig.tight_layout()
        fig.savefig(output_dir / "moe_gain.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    efficiency_df = df[df["average_fusion_makespan"].notna()].dropna(subset=["merged_unique_candidate_count"]).copy()
    if not efficiency_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        plot_bar(axes[0], efficiency_df, "experiment", "merged_unique_candidate_count", "Unique candidates after dedup", "Count")
        plot_bar(axes[1], efficiency_df, "experiment", "merged_committee_unique_utilization_ratio", "Committee utilization", "Utilization ratio")
        plot_bar(axes[2], efficiency_df, "experiment", "mean_committee_diversity", "Committee diversity", "Score")
        fig.tight_layout()
        fig.savefig(output_dir / "efficiency_main.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot and tabulate experiment summaries.")
    parser.add_argument("--output-dir", type=str, default="logs/figures", help="Directory to save figures and tables.")
    parser.add_argument(
        "--experiment",
        action="append",
        nargs=2,
        metavar=("LABEL", "SUMMARY_JSON"),
        help="Add experiment summary as label/path pair. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.experiment:
        experiments = {label: Path(path) for label, path in args.experiment}
    else:
        experiments = DEFAULT_EXPERIMENTS

    df = build_summary_frame(experiments)
    output_dir = Path(args.output_dir)
    save_tables(df, output_dir)
    save_plots(df, output_dir)
    df.to_csv(output_dir / "all_metrics.csv", index=False)
    print(f"[INFO] Figures and tables saved under: {output_dir}")


if __name__ == "__main__":
    main()

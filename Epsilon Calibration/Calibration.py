#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuration
BASE_DIR = Path(".")

OUTPUT_DIR = BASE_DIR / "outputs" / "epsilon_calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_HISTOGRAM_CSV = (
    BASE_DIR
    / "data"
    / "baselines"
    / "dataset_sensitive_word_length_histograms.csv"
)

K_ALPHABET = 94

FLOAT_TOL = 1e-12

SAVE_PLOTS = True

SETTINGS: dict[str, dict[str, Any]] = {
    "Health + GPT-4o mini": {
        "dataset": "Health",
        "model": "GPT-4o mini",
        "extracted_csv": (
            BASE_DIR
            / "data"
            / "extracted"
            / "health_gpt4o_mini_exact_reconstruction.csv"
        ),
        "utility_csv": (
            BASE_DIR
            / "data"
            / "utility"
            / "health_gpt4o_mini_summary_utility.csv"
        ),
        "subset_note": "",
    },
    "Health + Llama-3.1-8B": {
        "dataset": "Health",
        "model": "Llama-3.1-8B",
        "extracted_csv": (
            BASE_DIR
            / "data"
            / "extracted"
            / "health_llama31_8b_exact_reconstruction.csv"
        ),
        "utility_csv": (
            BASE_DIR
            / "data"
            / "utility"
            / "health_llama31_8b_summary_utility.csv"
        ),
        "subset_note": "",
    },
    "Health + GPT-5.4": {
        "dataset": "Health",
        "model": "GPT-5.4",
        "extracted_csv": (
            BASE_DIR
            / "data"
            / "extracted"
            / "health_gpt54_exact_reconstruction.csv"
        ),
        "utility_csv": (
            BASE_DIR
            / "data"
            / "utility"
            / "health_gpt54_summary_utility.csv"
        ),
        "subset_note": "1,000-prompt subset",
    },
    "Enron + GPT-4o mini": {
        "dataset": "Enron",
        "model": "GPT-4o mini",
        "extracted_csv": (
            BASE_DIR
            / "data"
            / "extracted"
            / "enron_gpt4o_mini_exact_reconstruction.csv"
        ),
        "utility_csv": (
            BASE_DIR
            / "data"
            / "utility"
            / "enron_gpt4o_mini_summary_utility.csv"
        ),
        "subset_note": "",
    },
    "Enron + Llama-3.1-8B": {
        "dataset": "Enron",
        "model": "Llama-3.1-8B",
        "extracted_csv": (
            BASE_DIR
            / "data"
            / "extracted"
            / "enron_llama31_8b_exact_reconstruction.csv"
        ),
        "utility_csv": (
            BASE_DIR
            / "data"
            / "utility"
            / "enron_llama31_8b_summary_utility.csv"
        ),
        "subset_note": "",
    },
}

EPSILON_COLUMN_CANDIDATES = [
    "epsilon",
    "Epsilon",
    "eps",
]

UTILITY_COLUMN_CANDIDATES = [
    "semantic_pct",
    "semantic_utility_pct",
    "average_semantic_similarity",
    "avg_semantic_similarity",
    "summary_semantic_score",
    "semantic_similarity",
    "semantic_score",
]

_BASELINE_CACHE: dict[str, Counter] = {}

# Validation

def require_file(
    path_value: str | Path,
    description: str,
) -> Path:

    path = Path(path_value)

    if not path.is_file():
        raise FileNotFoundError(
            f"{description} was not found:\n{path.resolve()}"
        )

    return path

def find_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
    description: str,
) -> str:

    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    raise ValueError(
        f"Could not find {description}.\n"
        f"Tried: {list(candidates)}\n"
        f"Available columns: {list(dataframe.columns)}"
    )

def require_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    description: str,
) -> None:

    missing = [
        column
        for column in columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{description} is missing required columns:\n"
            f"{missing}\n\n"
            f"Available columns:\n"
            f"{list(dataframe.columns)}"
        )

def safe_slug(text: str) -> str:

    return re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        text,
    ).strip("_").lower()

# Theoretical baseline

def load_dataset_sensitive_word_histogram(
    dataset: str,
    histogram_csv: Path = BASELINE_HISTOGRAM_CSV,
) -> Counter:

    cache_key = dataset.strip().casefold()

    if cache_key in _BASELINE_CACHE:
        return _BASELINE_CACHE[cache_key]

    path = require_file(
        histogram_csv,
        "Dataset-sensitive-word histogram CSV",
    )

    dataframe = pd.read_csv(
        path,
        low_memory=False,
    )

    require_columns(
        dataframe,
        [
            "dataset",
            "word_length",
            "word_count",
        ],
        "Dataset-sensitive-word histogram CSV",
    )

    subset = dataframe[
        dataframe["dataset"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .eq(cache_key)
    ].copy()

    if subset.empty:
        raise ValueError(
            f"No histogram rows were found for dataset={dataset!r} "
            f"in:\n{path}"
        )

    subset["word_length"] = pd.to_numeric(
        subset["word_length"],
        errors="coerce",
    )

    subset["word_count"] = pd.to_numeric(
        subset["word_count"],
        errors="coerce",
    )

    subset = subset.dropna(
        subset=[
            "word_length",
            "word_count",
        ]
    )

    subset = subset[
        (subset["word_length"] > 0)
        & (subset["word_count"] > 0)
    ]

    histogram: Counter = Counter()

    for _, row in subset.iterrows():
        word_length = int(row["word_length"])
        word_count = int(row["word_count"])
        histogram[word_length] += word_count

    if not histogram:
        raise ValueError(
            f"The sensitive-word histogram for {dataset} is empty."
        )

    total_words = int(sum(histogram.values()))

    mean_length = (
        sum(
            word_length * word_count
            for word_length, word_count in histogram.items()
        )
        / total_words
    )

    print("\n" + "=" * 80)
    print(f"{dataset.upper()} THEORETICAL BASELINE")
    print("=" * 80)
    print(f"Histogram source: {path}")
    print(f"Sensitive mechanism-words: {total_words:,}")
    print(f"Mean sensitive-word length: {mean_length:.4f}")
    print(
        "Sensitive-word length range: "
        f"{min(histogram)} to {max(histogram)}"
    )

    _BASELINE_CACHE[cache_key] = histogram
    return histogram

def retain_probability(
    epsilon: np.ndarray | float,
    k: int = K_ALPHABET,
) -> np.ndarray:

    epsilon_array = np.asarray(
        epsilon,
        dtype=float,
    )

    return 1.0 / (
        1.0
        + (k - 1) * np.exp(-epsilon_array)
    )

def theoretical_baseline_pct(
    epsilon: np.ndarray | float,
    length_histogram: Counter,
    k: int = K_ALPHABET,
) -> np.ndarray:

    epsilon_array = np.asarray(
        epsilon,
        dtype=float,
    )

    retain = retain_probability(
        epsilon_array,
        k=k,
    )

    total_words = int(
        sum(length_histogram.values())
    )

    if total_words <= 0:
        raise ValueError(
            "The sensitive-word histogram contains no words."
        )

    baseline_probability = np.zeros_like(
        epsilon_array,
        dtype=float,
    )

    for word_length, word_count in length_histogram.items():
        baseline_probability += (
            word_count / total_words
        ) * (
            retain ** int(word_length)
        )

    return 100.0 * baseline_probability

# Reconstruction

def load_reconstruction_curve(
    setting_name: str,
    config: dict[str, Any],
) -> pd.DataFrame:

    path = require_file(
        config["extracted_csv"],
        f"{setting_name} extracted CSV",
    )

    dataframe = pd.read_csv(
        path,
        low_memory=False,
    )

    epsilon_column = find_column(
        dataframe,
        EPSILON_COLUMN_CANDIDATES,
        f"epsilon column in {setting_name} extracted CSV",
    )

    if epsilon_column != "epsilon":
        dataframe = dataframe.rename(
            columns={
                epsilon_column: "epsilon",
            }
        )

    required_columns = [
        "epsilon",
        "original_total_sensitive_count",
        "total_matching_sensitive_count",
    ]

    require_columns(
        dataframe,
        required_columns,
        f"{setting_name} extracted CSV",
    )

    for column in required_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe = dataframe.dropna(
        subset=["epsilon"]
    ).copy()

    dataframe["original_total_sensitive_count"] = (
        dataframe["original_total_sensitive_count"]
        .fillna(0)
        .clip(lower=0)
    )

    dataframe["total_matching_sensitive_count"] = (
        dataframe["total_matching_sensitive_count"]
        .fillna(0)
        .clip(lower=0)
    )

    invalid_rows = (
        dataframe["total_matching_sensitive_count"]
        > dataframe["original_total_sensitive_count"]
    )

    if invalid_rows.any():
        examples = dataframe.loc[
            invalid_rows,
            [
                "epsilon",
                "original_total_sensitive_count",
                "total_matching_sensitive_count",
            ],
        ].head(10)

        raise ValueError(
            f"{setting_name}: reconstructed counts exceed original counts "
            f"in one or more rows.\n\n"
            f"Example rows:\n"
            f"{examples.to_string(index=False)}"
        )

    grouped = (
        dataframe.groupby(
            "epsilon",
            as_index=False,
        )
        .agg(
            num_prompts=(
                "epsilon",
                "size",
            ),
            original_sensitive_entities=(
                "original_total_sensitive_count",
                "sum",
            ),
            reconstructed_sensitive_entities=(
                "total_matching_sensitive_count",
                "sum",
            ),
        )
        .sort_values("epsilon")
        .reset_index(drop=True)
    )

    invalid_epsilons = grouped[
        grouped["original_sensitive_entities"] <= 0
    ]

    if not invalid_epsilons.empty:
        epsilon_values = invalid_epsilons[
            "epsilon"
        ].tolist()

        raise ValueError(
            f"{setting_name}: no original sensitive entities were found "
            f"at epsilon values {epsilon_values}."
        )

    grouped["exact_reconstruction_pct"] = (
        100.0
        * grouped["reconstructed_sensitive_entities"]
        / grouped["original_sensitive_entities"]
    )

    histogram = load_dataset_sensitive_word_histogram(
        config["dataset"]
    )

    grouped["theoretical_baseline_pct"] = theoretical_baseline_pct(
        grouped["epsilon"].to_numpy(),
        histogram,
    )

    grouped.insert(
        0,
        "setting",
        setting_name,
    )

    grouped.insert(
        1,
        "dataset",
        config["dataset"],
    )

    grouped.insert(
        2,
        "model",
        config["model"],
    )

    grouped.insert(
        3,
        "subset_note",
        config.get("subset_note", ""),
    )

    return grouped

# Utility

def find_utility_column(
    dataframe: pd.DataFrame,
    setting_name: str,
) -> str:

    for candidate in UTILITY_COLUMN_CANDIDATES:
        if candidate in dataframe.columns:
            return candidate

    raise ValueError(
        f"{setting_name}: unable to find a semantic utility column.\n"
        f"Tried: {UTILITY_COLUMN_CANDIDATES}\n"
        f"Available columns: {list(dataframe.columns)}"
    )

def convert_utility_to_percentage(
    series: pd.Series,
    setting_name: str,
) -> pd.Series:

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid = numeric.dropna()

    if valid.empty:
        raise ValueError(
            f"{setting_name}: utility column contains no numeric values."
        )

    minimum = float(valid.min())
    maximum = float(valid.max())

    if minimum >= -1.00001 and maximum <= 1.00001:
        return (
            100.0
            * numeric.clip(
                lower=-1.0,
                upper=1.0,
            )
        )

    if minimum >= -100.0001 and maximum <= 100.0001:
        return numeric.clip(
            lower=-100.0,
            upper=100.0,
        )

    raise ValueError(
        f"{setting_name}: utility values have an unexpected range "
        f"[{minimum}, {maximum}]."
    )

def load_utility_curve(
    setting_name: str,
    config: dict[str, Any],
) -> pd.DataFrame:

    path = require_file(
        config["utility_csv"],
        f"{setting_name} utility CSV",
    )

    dataframe = pd.read_csv(
        path,
        low_memory=False,
    )

    epsilon_column = find_column(
        dataframe,
        EPSILON_COLUMN_CANDIDATES,
        f"epsilon column in {setting_name} utility CSV",
    )

    if epsilon_column != "epsilon":
        dataframe = dataframe.rename(
            columns={
                epsilon_column: "epsilon",
            }
        )

    utility_column = find_utility_column(
        dataframe,
        setting_name,
    )

    dataframe["epsilon"] = pd.to_numeric(
        dataframe["epsilon"],
        errors="coerce",
    )

    dataframe["semantic_utility_pct"] = (
        convert_utility_to_percentage(
            dataframe[utility_column],
            setting_name,
        )
    )

    dataframe = dataframe.dropna(
        subset=[
            "epsilon",
            "semantic_utility_pct",
        ]
    ).copy()

    if dataframe.empty:
        raise ValueError(
            f"{setting_name}: no valid utility rows remain."
        )

    return (
        dataframe.groupby(
            "epsilon",
            as_index=False,
        )
        .agg(
            semantic_utility_pct=(
                "semantic_utility_pct",
                "mean",
            ),
            utility_source_rows=(
                "semantic_utility_pct",
                "size",
            ),
        )
        .sort_values("epsilon")
        .reset_index(drop=True)
    )

# Data merge

def merge_setting_data(
    setting_name: str,
    config: dict[str, Any],
) -> pd.DataFrame:

    reconstruction = load_reconstruction_curve(
        setting_name,
        config,
    )

    utility = load_utility_curve(
        setting_name,
        config,
    )

    merged = reconstruction.merge(
        utility,
        on="epsilon",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    unmatched = merged[
        merged["_merge"] != "both"
    ][
        [
            "epsilon",
            "_merge",
        ]
    ]

    if not unmatched.empty:
        print(
            f"\nWarning: unmatched epsilon values for {setting_name}:"
        )
        print(
            unmatched.to_string(index=False)
        )

    merged = (
        merged[
            merged["_merge"] == "both"
        ]
        .drop(columns="_merge")
        .sort_values("epsilon")
        .reset_index(drop=True)
    )

    if merged.empty:
        raise ValueError(
            f"{setting_name}: no common epsilon values were found "
            f"between reconstruction and utility data."
        )

    return merged

# Pareto frontier

def mark_pareto_frontier(
    dataframe: pd.DataFrame,
) -> pd.Series:

    frontier = pd.Series(
        True,
        index=dataframe.index,
        dtype=bool,
    )

    for index_i, row_i in dataframe.iterrows():
        for index_j, row_j in dataframe.iterrows():
            if index_i == index_j:
                continue

            at_least_as_good = (
                row_j["semantic_utility_pct"]
                >= row_i["semantic_utility_pct"] - FLOAT_TOL
                and
                row_j["conservative_privacy_pct"]
                >= row_i["conservative_privacy_pct"] - FLOAT_TOL
            )

            strictly_better = (
                row_j["semantic_utility_pct"]
                > row_i["semantic_utility_pct"] + FLOAT_TOL
                or
                row_j["conservative_privacy_pct"]
                > row_i["conservative_privacy_pct"] + FLOAT_TOL
            )

            if at_least_as_good and strictly_better:
                frontier.loc[index_i] = False
                break

    return frontier

# Calibration

def calibrate_setting(
    setting_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if setting_df.empty:
        raise ValueError(
            "Cannot calibrate an empty setting DataFrame."
        )

    output = (
        setting_df
        .sort_values("epsilon")
        .reset_index(drop=True)
        .copy()
    )

    setting_name = str(
        output.iloc[0]["setting"]
    )

    output["conservative_reconstruction_risk_pct"] = np.maximum(
        output["exact_reconstruction_pct"],
        output["theoretical_baseline_pct"],
    )

    output["conservative_privacy_pct"] = (
        100.0
        - output["conservative_reconstruction_risk_pct"]
    ).clip(
        lower=0.0,
        upper=100.0,
    )

    output["pareto_frontier"] = mark_pareto_frontier(
        output
    )

    output["distance_to_ideal"] = np.sqrt(
        (
            (
                100.0
                - output["semantic_utility_pct"]
            )
            / 100.0
        ) ** 2
        +
        (
            (
                100.0
                - output["conservative_privacy_pct"]
            )
            / 100.0
        ) ** 2
    )

    pareto = output[
        output["pareto_frontier"]
    ].copy()

    if pareto.empty:
        raise ValueError(
            f"{setting_name}: no Pareto-efficient points were found."
        )

    pareto = pareto.sort_values(
        by=[
            "distance_to_ideal",
            "semantic_utility_pct",
            "conservative_privacy_pct",
            "epsilon",
        ],
        ascending=[
            True,
            False,
            False,
            True,
        ],
    )

    selected_index = int(
        pareto.index[0]
    )

    output["selected_calibrated_point"] = False

    output.loc[
        selected_index,
        "selected_calibrated_point",
    ] = True

    selected = output.loc[selected_index]

    selected_row = pd.DataFrame(
        [
            {
                "setting": selected["setting"],
                "dataset": selected["dataset"],
                "model": selected["model"],
                "subset_note": selected["subset_note"],
                "selected_epsilon": float(
                    selected["epsilon"]
                ),
                "semantic_utility_pct": float(
                    selected["semantic_utility_pct"]
                ),
                "exact_reconstruction_pct": float(
                    selected["exact_reconstruction_pct"]
                ),
                "theoretical_baseline_pct": float(
                    selected["theoretical_baseline_pct"]
                ),
                "conservative_reconstruction_risk_pct": float(
                    selected[
                        "conservative_reconstruction_risk_pct"
                    ]
                ),
                "conservative_privacy_pct": float(
                    selected["conservative_privacy_pct"]
                ),
                "distance_to_ideal": float(
                    selected["distance_to_ideal"]
                ),
                "num_prompts": int(
                    selected["num_prompts"]
                ),
                "original_sensitive_entities": int(
                    selected["original_sensitive_entities"]
                ),
                "reconstructed_sensitive_entities": int(
                    selected["reconstructed_sensitive_entities"]
                ),
                "selection_rule": (
                    "pareto_efficient_point_closest_to_"
                    "100pct_utility_and_100pct_privacy"
                ),
            }
        ]
    )

    return output, selected_row

# Plotting

def plot_setting(
    setting_df: pd.DataFrame,
    output_path: Path,
) -> None:

    setting_df = setting_df.sort_values(
        "conservative_privacy_pct"
    )

    pareto = setting_df[
        setting_df["pareto_frontier"]
    ].sort_values(
        "conservative_privacy_pct"
    )

    selected = setting_df[
        setting_df["selected_calibrated_point"]
    ]

    fig, axis = plt.subplots(
        figsize=(7.8, 5.8)
    )

    axis.scatter(
        setting_df["conservative_privacy_pct"],
        setting_df["semantic_utility_pct"],
        s=48,
        alpha=0.65,
        label=r"Evaluated $\epsilon$ values",
    )

    if not pareto.empty:
        axis.plot(
            pareto["conservative_privacy_pct"],
            pareto["semantic_utility_pct"],
            marker="o",
            linewidth=1.8,
            label="Pareto frontier",
        )

    for _, row in setting_df.iterrows():
        axis.annotate(
            f"{row['epsilon']:g}",
            (
                row["conservative_privacy_pct"],
                row["semantic_utility_pct"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    if not selected.empty:
        row = selected.iloc[0]

        axis.scatter(
            [row["conservative_privacy_pct"]],
            [row["semantic_utility_pct"]],
            marker="*",
            s=230,
            zorder=6,
            label=(
                rf"Selected $\epsilon={row['epsilon']:g}$"
            ),
        )

        axis.plot(
            [
                row["conservative_privacy_pct"],
                100.0,
            ],
            [
                row["semantic_utility_pct"],
                100.0,
            ],
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
        )

    axis.scatter(
        [100.0],
        [100.0],
        marker="X",
        s=100,
        label="Ideal point (100%, 100%)",
        zorder=5,
    )

    axis.set_xlabel(
        "Conservative privacy preservation (%)"
    )

    axis.set_ylabel(
        "Downstream semantic utility (%)"
    )

    axis.set_title(
        str(setting_df.iloc[0]["setting"])
    )

    axis.set_xlim(
        0,
        102,
    )

    axis.set_ylim(
        0,
        102,
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    axis.legend(
        fontsize=8,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

def plot_combined(
    all_points: pd.DataFrame,
    output_path: Path,
) -> None:

    fig, axis = plt.subplots(
        figsize=(10, 7)
    )

    for setting_name, setting_df in all_points.groupby(
        "setting",
        sort=False,
    ):
        pareto = setting_df[
            setting_df["pareto_frontier"]
        ].sort_values(
            "conservative_privacy_pct"
        )

        axis.plot(
            pareto["conservative_privacy_pct"],
            pareto["semantic_utility_pct"],
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=setting_name,
        )

        selected = setting_df[
            setting_df["selected_calibrated_point"]
        ]

        if not selected.empty:
            axis.scatter(
                selected["conservative_privacy_pct"],
                selected["semantic_utility_pct"],
                marker="*",
                s=140,
                zorder=6,
            )

    axis.scatter(
        [100.0],
        [100.0],
        marker="X",
        s=110,
        label="Ideal point (100%, 100%)",
        zorder=5,
    )

    axis.set_xlabel(
        "Conservative privacy preservation (%)"
    )

    axis.set_ylabel(
        "Downstream semantic utility (%)"
    )

    axis.set_title(
        "Model-specific baseline-aware privacy--utility calibration"
    )

    axis.set_xlim(
        0,
        102,
    )

    axis.set_ylim(
        0,
        102,
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    axis.legend(
        fontsize=8,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

# Output tables

def build_compact_table(
    selected_df: pd.DataFrame,
) -> pd.DataFrame:

    selected = selected_df.copy()

    selected["paper_entry"] = selected.apply(
        lambda row: (
            f"{row['selected_epsilon']:.1f} "
            f"(U={row['semantic_utility_pct']:.2f}%, "
            f"P={row['conservative_privacy_pct']:.2f}%, "
            f"R={row['exact_reconstruction_pct']:.3f}%)"
        ),
        axis=1,
    )

    compact = selected.pivot(
        index="dataset",
        columns="model",
        values="paper_entry",
    )

    compact = compact.reindex(
        index=[
            "Health",
            "Enron",
        ],
        columns=[
            "GPT-4o mini",
            "Llama-3.1-8B",
            "GPT-5.4",
        ],
    )

    return (
        compact
        .fillna("--")
        .reset_index()
    )

def save_latex_table(
    selected_df: pd.DataFrame,
    output_path: Path,
) -> None:

    values = {
        (
            str(row["dataset"]),
            str(row["model"]),
        ): row
        for _, row in selected_df.iterrows()
    }

    def entry(
        dataset: str,
        model: str,
    ) -> str:
        row = values.get(
            (
                dataset,
                model,
            )
        )

        if row is None:
            return "--"

        return (
            rf"\textbf{{{row['selected_epsilon']:.1f}}} "
            rf"({row['semantic_utility_pct']:.2f}\%, "
            rf"{row['conservative_privacy_pct']:.2f}\%)"
        )

    latex = rf"""
\begin{{table}}[t]
\centering
\caption{{Model-specific baseline-aware privacy--utility calibration results.}}
\label{{tab:balanced-model-calibration}}
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lccc}}
\toprule
Dataset & GPT-4o mini & Llama-3.1-8B & GPT-5.4$^{{\dagger}}$ \\
\midrule
Health &
{entry('Health', 'GPT-4o mini')} &
{entry('Health', 'Llama-3.1-8B')} &
{entry('Health', 'GPT-5.4')} \\
Enron &
{entry('Enron', 'GPT-4o mini')} &
{entry('Enron', 'Llama-3.1-8B')} &
-- \\
\bottomrule
\end{{tabular}}%
}}
\vspace{{1mm}}

\footnotesize{{Each entry reports the selected $\epsilon^*$ followed by
(semantic utility, conservative privacy preservation). Conservative privacy
preservation is defined as 100\% minus the larger of the empirical exact
sensitive-entity reconstruction rate and the dataset-specific theoretical
reconstruction baseline. The selected point is the Pareto-efficient point
closest to the ideal of 100\% utility and 100\% privacy.
$^{{\dagger}}$GPT-5.4 uses the 1{{,}}000-prompt Health subset.}}
\end{{table}}
""".strip()

    output_path.write_text(
        latex,
        encoding="utf-8",
    )

# Sanity checks

def run_sanity_checks(
    all_points: pd.DataFrame,
    selected_points: pd.DataFrame,
) -> None:

    if len(selected_points) != len(SETTINGS):
        raise AssertionError(
            "Expected exactly one selected epsilon per setting, "
            f"but found {len(selected_points)} selections for "
            f"{len(SETTINGS)} settings."
        )

    selected_source_rows = all_points[
        all_points["selected_calibrated_point"]
    ]

    if len(selected_source_rows) != len(SETTINGS):
        raise AssertionError(
            "Expected exactly one selected row per setting in the "
            "detailed output."
        )

    if not selected_source_rows[
        "pareto_frontier"
    ].all():
        raise AssertionError(
            "At least one selected epsilon is not Pareto-efficient."
        )

    expected_risk = np.maximum(
        all_points["exact_reconstruction_pct"],
        all_points["theoretical_baseline_pct"],
    )

    if not np.allclose(
        all_points["conservative_reconstruction_risk_pct"],
        expected_risk,
        rtol=0.0,
        atol=FLOAT_TOL,
    ):
        raise AssertionError(
            "The conservative reconstruction risk was not calculated "
            "using max(empirical reconstruction, theoretical baseline)."
        )

    expected_privacy = (
        100.0
        - expected_risk
    ).clip(
        lower=0.0,
        upper=100.0,
    )

    if not np.allclose(
        all_points["conservative_privacy_pct"],
        expected_privacy,
        rtol=0.0,
        atol=FLOAT_TOL,
    ):
        raise AssertionError(
            "The conservative privacy-preserved percentage is incorrect."
        )

    for setting_name, setting_df in all_points.groupby(
        "setting",
        sort=False,
    ):
        selected_rows = setting_df[
            setting_df["selected_calibrated_point"]
        ]

        if len(selected_rows) != 1:
            raise AssertionError(
                f"{setting_name}: expected exactly one selected row."
            )

        selected_distance = float(
            selected_rows["distance_to_ideal"].iloc[0]
        )

        best_pareto_distance = float(
            setting_df.loc[
                setting_df["pareto_frontier"],
                "distance_to_ideal",
            ].min()
        )

        if not math.isclose(
            selected_distance,
            best_pareto_distance,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise AssertionError(
                f"{setting_name}: selected point is not the "
                f"Pareto-efficient point closest to the ideal."
            )

    print("\nSanity checks passed.")

# Main

def main() -> None:

    print("=" * 80)
    print("BASELINE-AWARE PRIVACY--UTILITY PARETO CALIBRATION")
    print("=" * 80)
    print(f"Character alphabet size: {K_ALPHABET}")
    print(
        "Conservative risk: "
        "max(empirical reconstruction, theoretical baseline)"
    )
    print(
        "Conservative privacy: 100% - conservative risk"
    )
    print(
        "Selection: Pareto-efficient point closest to "
        "(100% utility, 100% privacy)"
    )
    print(
        "Objective weighting: equal"
    )
    print(
        "Per-setting min--max normalisation: disabled"
    )
    print(
        f"Baseline histogram: {BASELINE_HISTOGRAM_CSV}"
    )
    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    all_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []

    for setting_name, config in SETTINGS.items():
        print("\n" + "=" * 80)
        print(f"Processing: {setting_name}")
        print("=" * 80)

        merged = merge_setting_data(
            setting_name,
            config,
        )

        calibrated, selected = calibrate_setting(
            merged
        )

        all_frames.append(
            calibrated
        )

        selected_frames.append(
            selected
        )

        selected_row = selected.iloc[0]

        print(
            f"Selected epsilon: "
            f"{selected_row['selected_epsilon']:.1f}"
        )

        print(
            "Semantic utility: "
            f"{selected_row['semantic_utility_pct']:.4f}%"
        )

        print(
            "Empirical exact reconstruction: "
            f"{selected_row['exact_reconstruction_pct']:.6f}%"
        )

        print(
            "Theoretical baseline: "
            f"{selected_row['theoretical_baseline_pct']:.6f}%"
        )

        print(
            "Conservative reconstruction risk: "
            f"{selected_row['conservative_reconstruction_risk_pct']:.6f}%"
        )

        print(
            "Conservative privacy preservation: "
            f"{selected_row['conservative_privacy_pct']:.6f}%"
        )

        print(
            "Distance to ideal: "
            f"{selected_row['distance_to_ideal']:.8f}"
        )

        if SAVE_PLOTS:
            plot_setting(
                calibrated,
                OUTPUT_DIR
                / (
                    f"{safe_slug(setting_name)}"
                    f"_privacy_utility_calibration.png"
                ),
            )

    all_points = pd.concat(
        all_frames,
        ignore_index=True,
    )

    selected_points = pd.concat(
        selected_frames,
        ignore_index=True,
    )

    selected_points = (
        selected_points
        .sort_values(
            [
                "dataset",
                "model",
            ]
        )
        .reset_index(drop=True)
    )

    pareto_points = all_points[
        all_points["pareto_frontier"]
    ].copy()

    compact_table = build_compact_table(
        selected_points
    )

    all_points_path = (
        OUTPUT_DIR
        / "all_calibration_points.csv"
    )

    pareto_points_path = (
        OUTPUT_DIR
        / "pareto_frontier_points.csv"
    )

    selected_points_path = (
        OUTPUT_DIR
        / "selected_calibrated_epsilons.csv"
    )

    compact_table_path = (
        OUTPUT_DIR
        / "selected_calibrated_epsilons_compact.csv"
    )

    latex_table_path = (
        OUTPUT_DIR
        / "selected_calibrated_epsilons_latex.tex"
    )

    combined_plot_path = (
        OUTPUT_DIR
        / "combined_privacy_utility_calibration.png"
    )

    all_points.to_csv(
        all_points_path,
        index=False,
    )

    pareto_points.to_csv(
        pareto_points_path,
        index=False,
    )

    selected_points.to_csv(
        selected_points_path,
        index=False,
    )

    compact_table.to_csv(
        compact_table_path,
        index=False,
    )

    save_latex_table(
        selected_points,
        latex_table_path,
    )

    if SAVE_PLOTS:
        plot_combined(
            all_points,
            combined_plot_path,
        )

    run_sanity_checks(
        all_points,
        selected_points,
    )

    print("\n" + "=" * 80)
    print("SELECTED CALIBRATED OPERATING POINTS")
    print("=" * 80)

    print(
        selected_points[
            [
                "dataset",
                "model",
                "selected_epsilon",
                "semantic_utility_pct",
                "exact_reconstruction_pct",
                "theoretical_baseline_pct",
                "conservative_reconstruction_risk_pct",
                "conservative_privacy_pct",
                "distance_to_ideal",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nSaved outputs:")
    print(f"  All evaluated points: {all_points_path}")
    print(f"  Pareto points:        {pareto_points_path}")
    print(f"  Selected points:      {selected_points_path}")
    print(f"  Compact table:        {compact_table_path}")
    print(f"  LaTeX table:          {latex_table_path}")

    if SAVE_PLOTS:
        print(f"  Combined plot:        {combined_plot_path}")

if __name__ == "__main__":
    main()
"""
Plotting utility functions for test-time scaling analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_NAME_MAPPING = {
    "RFT": "RFT",
    "simplescaling/s1.1-7B": "s1.1-7B",
    "open-thoughts/OpenThinker3-7B": "OpenThinker3-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "DeepSeek-R1-7B",
    "Qwen2.5 7B Instruct": "Qwen2.5 7B Instruct",
    "mistralai/Ministral-8B-Instruct-2410": "Ministral 8B Instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1 8B Instruct",
    "google/gemma-3-4b-it": "Gemma 3 4B Instruct",
}

DEFAULT_BENCHMARKS = ["AIME 2025", "MATH500", "MMLU Pro-1K", "SuperGPQA-1K"]


def prepare_display_dataframe(
    df: pd.DataFrame,
    benchmarks: list[str] = None,
    model_mapping: dict[str, str] = None,
    apply_styling: bool = True,
):
    """
    Prepare and optionally style a dataframe for display.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw results dataframe with Model, Prompting, and benchmark columns
    benchmarks : list[str], optional
        List of benchmark column names. Defaults to DEFAULT_BENCHMARKS
    model_mapping : dict[str, str], optional
        Mapping of original model names to display names. Defaults to MODEL_NAME_MAPPING
    apply_styling : bool, optional
        Whether to apply styling to the dataframe. Defaults to True

    Returns:
    --------
    pd.DataFrame or pd.io.formats.style.Styler
        Prepared dataframe, optionally styled
    """
    if benchmarks is None:
        benchmarks = DEFAULT_BENCHMARKS
    if model_mapping is None:
        model_mapping = MODEL_NAME_MAPPING

    display_df = df.copy()
    display_df["Model"] = display_df["Model"].map(model_mapping)
    display_df["Approach"] = display_df["Prompting"].map(
        lambda x: "Zero-shot" if x in ["Zero-shot", "CoT"] else "CoT+BF"
    )
    display_df = display_df[["Model", "Approach"] + benchmarks]

    # Group by Model and Approach to handle duplicates (e.g., both Zero-shot and CoT map to "Zero-shot")
    display_df = display_df.groupby(["Model", "Approach"], as_index=False)[
        benchmarks
    ].mean()
    display_df["Average"] = display_df[benchmarks].mean(axis=1)

    if not apply_styling:
        return display_df

    styled_table = (
        display_df.style.format(
            {**{bench: "{:.2f}" for bench in benchmarks}, "Average": "{:.2f}"}
        )
        .background_gradient(
            subset=benchmarks + ["Average"], cmap="RdYlGn", vmin=0, vmax=100
        )
        .set_properties(
            **{"text-align": "center", "font-size": "11pt", "border": "1px solid #ddd"}
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#2c3e50"),
                        ("color", "white"),
                        ("font-weight", "bold"),
                        ("text-align", "center"),
                        ("padding", "12px"),
                        ("font-size", "12pt"),
                    ],
                },
                {"selector": "td", "props": [("padding", "10px")]},
                {"selector": "tr:hover", "props": [("background-color", "#e8f4f8")]},
            ]
        )
    )

    return styled_table


def plot_performance_comparison(
    df: pd.DataFrame,
    benchmarks: list[str] = None,
    output_path: str = "outputs/fig_1_performance_comparison.png",
    figsize: tuple = (16, 12),
    dpi: int = 300,
) -> pd.DataFrame:
    """
    Create a grouped bar chart comparing model performance across benchmarks.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing Model, Approach, and benchmark columns
    benchmarks : list[str], optional
        List of benchmark column names. Defaults to ["AIME 2025", "MATH500", "MMLU Pro-1K", "SuperGPQA-1K"]
    output_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size as (width, height)
    dpi : int, optional
        Resolution for saved figure

    Returns:
    --------
    pd.DataFrame
        Copy of the input dataframe used for plotting
    """
    if benchmarks is None:
        benchmarks = ["AIME 2025", "MATH500", "MMLU Pro-1K", "SuperGPQA-1K"]

    plot_df = df.copy()
    models_list = plot_df["Model"].unique()
    num_models = len(models_list)

    # Dynamically determine subplot layout based on number of models
    if num_models == 1:
        nrows, ncols = 1, 1
        if figsize == (16, 12):  # If using default, adjust for single plot
            figsize = (10, 7)
    elif num_models == 2:
        nrows, ncols = 1, 2
        if figsize == (16, 12):
            figsize = (16, 6)
    elif num_models == 3:
        nrows, ncols = 1, 3
        if figsize == (16, 12):
            figsize = (18, 6)
    else:  # 4 or more models
        nrows = (num_models + 1) // 2
        ncols = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Handle axes as array or single axis
    if num_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = {"Zero-shot": "#3498db", "CoT+BF": "#e74c3c"}

    for idx, model in enumerate(models_list):
        ax = axes[idx]
        model_data = plot_df[plot_df["Model"] == model]

        x = np.arange(len(benchmarks))
        width = 0.35

        # Group by Approach and take mean to handle multiple rows per approach
        zero_shot_subset = model_data[model_data["Approach"] == "Zero-shot"]
        cot_bf_subset = model_data[model_data["Approach"] == "CoT+BF"]

        zero_shot_data = (
            zero_shot_subset[benchmarks].mean(axis=0).values
            if len(zero_shot_subset) > 0
            else np.zeros(len(benchmarks))
        )
        cot_bf_data = (
            cot_bf_subset[benchmarks].mean(axis=0).values
            if len(cot_bf_subset) > 0
            else np.zeros(len(benchmarks))
        )

        bars1 = ax.bar(
            x - width / 2,
            zero_shot_data,
            width,
            label="Zero-shot",
            color=colors["Zero-shot"],
            alpha=0.8,
            edgecolor="black",
            linewidth=1.2,
        )
        bars2 = ax.bar(
            x + width / 2,
            cot_bf_data,
            width,
            label="CoT+BF",
            color=colors["CoT+BF"],
            alpha=0.8,
            edgecolor="black",
            linewidth=1.2,
        )

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

        ax.set_xlabel("Benchmark", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.set_title(f"{model}", fontsize=14, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks, rotation=15, ha="right")
        ax.legend(loc="upper left", framealpha=0.9, fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_ylim(0, 100)

        for i, (zs, cb) in enumerate(zip(zero_shot_data, cot_bf_data)):
            if cb > zs:
                improvement = ((cb - zs) / zs) * 100
                ax.annotate(
                    f"↑{improvement:.0f}%",
                    xy=(i, max(zs, cb) + 2),
                    ha="center",
                    fontsize=8,
                    color="green",
                    fontweight="bold",
                )
            elif cb < zs:
                decline = ((zs - cb) / zs) * 100
                ax.annotate(
                    f"↓{decline:.0f}%",
                    xy=(i, max(zs, cb) + 2),
                    ha="center",
                    fontsize=8,
                    color="red",
                    fontweight="bold",
                )

    # Hide unused subplots if any
    total_subplots = nrows * ncols
    if num_models < total_subplots:
        for idx in range(num_models, total_subplots):
            axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.show()

    return plot_df


def plot_average_performance(
    df: pd.DataFrame,
    output_path: str = "outputs/fig_2_average_performance.png",
    figsize: tuple = (14, 7),
    dpi: int = 300,
) -> None:
    """
    Create a bar chart comparing average performance across models.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing Model, Approach, and Average columns
    output_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size as (width, height)
    dpi : int, optional
        Resolution for saved figure
    """
    models_list = df["Model"].unique()
    num_models = len(models_list)

    # Adjust figsize for single model if using default
    if num_models == 1 and figsize == (14, 7):
        figsize = (8, 6)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(num_models)

    # Dynamically adjust bar width based on number of models
    if num_models == 1:
        width = 0.15  # Narrower bars for single model
    elif num_models == 2:
        width = 0.25
    elif num_models == 3:
        width = 0.3
    else:
        width = 0.35  # Default width for 4+ models

    zero_shot_avg = [
        df[(df["Model"] == m) & (df["Approach"] == "Zero-shot")]["Average"].mean()
        if len(df[(df["Model"] == m) & (df["Approach"] == "Zero-shot")]) > 0
        else 0
        for m in models_list
    ]
    cot_bf_avg = [
        df[(df["Model"] == m) & (df["Approach"] == "CoT+BF")]["Average"].mean()
        if len(df[(df["Model"] == m) & (df["Approach"] == "CoT+BF")]) > 0
        else 0
        for m in models_list
    ]

    bars1 = ax.bar(
        x - width / 2,
        zero_shot_avg,
        width,
        label="Zero-shot",
        color="#3498db",
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        cot_bf_avg,
        width,
        label="CoT+BF",
        color="#e74c3c",
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

    for i, (zs, cb) in enumerate(zip(zero_shot_avg, cot_bf_avg)):
        diff = cb - zs
        color = "green" if diff > 0 else "red"
        symbol = "↑" if diff > 0 else "↓"
        ax.plot(
            [i - width / 2, i + width / 2],
            [max(zs, cb) + 3, max(zs, cb) + 3],
            "k-",
            linewidth=2,
            alpha=0.3,
        )
        ax.text(
            i,
            max(zs, cb) + 5,
            f"{symbol}{abs(diff):.1f}",
            ha="center",
            fontsize=11,
            color=color,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor=color,
                linewidth=2,
            ),
        )

    ax.set_xlabel("Model", fontsize=14, fontweight="bold")
    ax.set_ylabel("Average Score", fontsize=14, fontweight="bold")
    ax.set_xticks(x)

    # Adjust label rotation based on number of models
    if num_models == 1:
        ax.set_xticklabels(models_list, rotation=0, ha="center", fontsize=11)
    else:
        ax.set_xticklabels(models_list, rotation=15, ha="right", fontsize=11)

    ax.legend(loc="upper left", framealpha=0.9, fontsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, max(zero_shot_avg + cot_bf_avg) * 1.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.show()


def prepare_budget_scaling_dataframe(
    df: pd.DataFrame, benchmarks: list[str] = None, model_mapping: dict[str, str] = None
) -> pd.DataFrame:
    """
    Prepare dataframe for budget scaling analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw results dataframe with Model, Budget, and benchmark columns
    benchmarks : list[str], optional
        List of benchmark column names. Defaults to DEFAULT_BENCHMARKS
    model_mapping : dict[str, str], optional
        Mapping of original model names to display names

    Returns:
    --------
    pd.DataFrame
        Prepared dataframe with Model_Pretty and Average columns
    """
    if benchmarks is None:
        benchmarks = DEFAULT_BENCHMARKS
    if model_mapping is None:
        model_mapping = MODEL_NAME_MAPPING

    result_df = df.copy()
    result_df["Average"] = result_df[benchmarks].mean(axis=1, skipna=True)
    result_df["Model_Pretty"] = result_df["Model"].map(model_mapping)

    return result_df


def format_budget_table(
    df: pd.DataFrame, benchmarks: list[str] = None, model_mapping: dict[str, str] = None
) -> str:
    """
    Format a budget scaling table for display as markdown.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw results dataframe with Model, Budget, and benchmark columns
    benchmarks : list[str], optional
        List of benchmark column names. Defaults to DEFAULT_BENCHMARKS
    model_mapping : dict[str, str], optional
        Mapping of original model names to display names

    Returns:
    --------
    str
        Markdown formatted table
    """
    if benchmarks is None:
        benchmarks = DEFAULT_BENCHMARKS
    if model_mapping is None:
        model_mapping = MODEL_NAME_MAPPING

    table_df = df.copy()
    table_df["Model_Pretty"] = table_df["Model"].map(model_mapping)
    table_df["Average"] = table_df[benchmarks].mean(axis=1)

    def format_num_2f(x):
        return f"{float(x):.2f}"

    def format_budget(x):
        if float(x).is_integer():
            return str(int(x))
        return f"{float(x):.2f}"

    for col in benchmarks + ["Average"]:
        table_df[col] = table_df[col].apply(format_num_2f)

    table_df["Budget"] = table_df["Budget"].apply(format_budget)

    display_cols = ["Model_Pretty", "Budget"] + benchmarks + ["Average"]

    return table_df[display_cols].to_markdown(index=False, floatfmt=".2f")


def plot_average_score_vs_budget(
    df: pd.DataFrame,
    output_path: str = "outputs/fig_3_avg_score_linechart.png",
    figsize: tuple = (10, 7),
    dpi: int = 300,
    colors: list[str] = None,
) -> None:
    """
    Create a line plot showing average scores vs budget for each model.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with Model, Model_Pretty, Budget, and Average columns
    output_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size as (width, height)
    dpi : int, optional
        Resolution for saved figure
    colors : list[str], optional
        List of color codes for models
    """
    if colors is None:
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#e67e22", "#1abc9c"]

    models_list = df["Model"].unique()

    plt.figure(figsize=figsize)
    for idx, model in enumerate(models_list):
        model_df = df[df["Model"] == model]
        color = colors[idx % len(colors)]
        pretty_model_name = (
            model_df["Model_Pretty"].iloc[0] if len(model_df) > 0 else model
        )

        plt.plot(
            model_df["Budget"],
            model_df["Average"],
            marker="o",
            label=pretty_model_name,
            color=color,
            linewidth=2.5,
            alpha=0.8,
        )

        for x, y in zip(model_df["Budget"], model_df["Average"]):
            plt.text(
                x,
                y + 0.8,
                f"{y:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=color,
            )

    plt.xlabel("Budget (tokens)", fontsize=13, fontweight="bold")
    plt.ylabel("Average Score (%)", fontsize=13, fontweight="bold")
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.xscale("log", base=2)

    budget_vals = df["Budget"].dropna().sort_values().unique()
    plt.xticks(
        budget_vals,
        [
            r"$2^{%d}$" % int(np.log2(x))
            if x > 0 and np.log2(x).is_integer()
            else str(x)
            for x in budget_vals
        ],
        rotation=0,
        fontsize=11,
    )

    plt.yticks(fontsize=11)
    plt.legend(title="Model", fontsize=11, title_fontsize=12, framealpha=0.92)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_benchmark_subplots_vs_budget(
    df: pd.DataFrame,
    benchmarks: list[str] = None,
    output_path: str = "outputs/fig_4_benchmark_subplots.png",
    figsize: tuple = (16, 12),
    dpi: int = 300,
    colors: list[str] = None,
) -> None:
    """
    Create subplots showing performance vs budget for each benchmark.

    The function automatically analyzes the data range across all benchmarks
    and sets consistent y-axis limits for easier comparison. The limits are
    rounded to nice numbers (multiples of 5) with 15% padding.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with Model, Model_Pretty, Budget, and benchmark columns
    benchmarks : list[str], optional
        List of benchmark column names. Defaults to DEFAULT_BENCHMARKS
    output_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size as (width, height)
    dpi : int, optional
        Resolution for saved figure
    colors : list[str], optional
        List of color codes for models
    """
    if benchmarks is None:
        benchmarks = DEFAULT_BENCHMARKS
    if colors is None:
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#e67e22", "#1abc9c"]

    models_list = df["Model"].unique()

    # Analyze data range across all benchmarks for consistent y-axis
    all_values = []
    for benchmark in benchmarks:
        values = df[benchmark].dropna()
        all_values.extend(values.tolist())

    if all_values:
        data_min = min(all_values)
        data_max = max(all_values)
        data_range = data_max - data_min

        # Set y-axis limits with padding
        y_padding = data_range * 0.15  # 15% padding
        y_min = max(0, data_min - y_padding)  # Don't go below 0
        y_max = min(100, data_max + y_padding)  # Don't exceed 100 for percentages

        # Round to nice numbers
        y_min = np.floor(y_min / 5) * 5
        y_max = np.ceil(y_max / 5) * 5
    else:
        y_min, y_max = 0, 100

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for bench_idx, benchmark in enumerate(benchmarks):
        ax = axes[bench_idx]

        for model_idx, model in enumerate(models_list):
            model_df = df[df["Model"] == model].sort_values("Budget")
            color = colors[model_idx % len(colors)]
            pretty_model_name = (
                model_df["Model_Pretty"].iloc[0] if len(model_df) > 0 else model
            )

            ax.plot(
                model_df["Budget"],
                model_df[benchmark],
                marker="o",
                label=pretty_model_name,
                color=color,
                linewidth=2.5,
                alpha=0.8,
            )

            # Calculate label offset based on y-axis range
            label_offset = (y_max - y_min) * 0.02  # 2% of range

            for x, y in zip(model_df["Budget"], model_df[benchmark]):
                if pd.notna(y):
                    ax.text(
                        x,
                        y + label_offset,
                        f"{y:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                        color=color,
                        alpha=0.7,
                    )

        ax.set_xlabel("Budget (tokens)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
        ax.set_title(benchmark, fontsize=14, fontweight="bold", pad=15)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_xscale("log", base=2)

        # Set consistent y-axis limits across all subplots
        ax.set_ylim(y_min, y_max)

        budget_vals = df["Budget"].dropna().sort_values().unique()
        ax.set_xticks(budget_vals)
        ax.set_xticklabels(
            [
                r"$2^{%d}$" % int(np.log2(x))
                if x > 0 and np.log2(x).is_integer()
                else str(x)
                for x in budget_vals
            ],
            rotation=0,
            fontsize=10,
        )
        ax.tick_params(axis="y", labelsize=10)

        if bench_idx == 0:
            ax.legend(
                title="Model",
                fontsize=10,
                title_fontsize=11,
                framealpha=0.92,
                loc="best",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_keyword_comparison_by_model(
    df: pd.DataFrame,
    keywords: list[str] = None,
    model_mapping: dict[str, str] = None,
    output_path: str = "outputs/fig_keyword_comparison.png",
    figsize: tuple = (16, 12),
    dpi: int = 300,
) -> None:
    """
    Create subplots showing average performance for different keywords, one subplot per model.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with Model, Keyword, and Average columns
    keywords : list[str], optional
        List of keywords to compare. Defaults to ["Wait", "Perhaps", "Let"]
    model_mapping : dict[str, str], optional
        Mapping of original model names to display names. Defaults to MODEL_NAME_MAPPING
    output_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size as (width, height)
    dpi : int, optional
        Resolution for saved figure
    """
    if keywords is None:
        keywords = ["Wait", "Perhaps", "Let"]
    if model_mapping is None:
        model_mapping = MODEL_NAME_MAPPING

    models_list = df["Model"].unique()
    num_models = len(models_list)

    # Determine subplot layout
    if num_models <= 2:
        nrows, ncols = 1, num_models
        if figsize == (16, 12):
            figsize = (8 * num_models, 6)
    elif num_models <= 4:
        nrows, ncols = 2, 2
        if figsize == (16, 12):
            figsize = (14, 10)
    else:
        nrows = (num_models + 1) // 2
        ncols = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Handle single subplot case
    if num_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Calculate global y-axis range for consistency
    all_averages = df["Average"].dropna().values
    if len(all_averages) > 0:
        y_min = max(0, np.floor(all_averages.min() / 5) * 5 - 5)
        y_max = min(100, np.ceil(all_averages.max() / 5) * 5 + 5)
    else:
        y_min, y_max = 0, 100

    # Color scheme for keywords
    keyword_colors = {
        "Wait": "#3498db",
        "Perhaps": "#e74c3c",
        "Let": "#2ecc71",
    }

    for idx, model in enumerate(models_list):
        ax = axes[idx]
        model_data = df[df["Model"] == model]

        # Get average scores for each keyword
        keyword_scores = []
        keyword_labels = []
        colors_list = []

        for keyword in keywords:
            keyword_data = model_data[model_data["Keyword"] == keyword]
            if len(keyword_data) > 0:
                avg_score = keyword_data["Average"].mean()
                keyword_scores.append(avg_score)
                keyword_labels.append(keyword)
                colors_list.append(keyword_colors.get(keyword, "#95a5a6"))

        x = np.arange(len(keyword_labels))
        bars = ax.bar(
            x,
            keyword_scores,
            width=0.6,
            color=colors_list,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        )

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        # Get pretty model name
        pretty_name = model_mapping.get(model, model)
        ax.set_title(pretty_name, fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Keyword", fontsize=12, fontweight="bold")
        ax.set_ylabel("Average Score (%)", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(keyword_labels, fontsize=11)
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.tick_params(axis="y", labelsize=10)

    # Hide unused subplots if any
    total_subplots = nrows * ncols
    if num_models < total_subplots:
        for idx in range(num_models, total_subplots):
            axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.show()

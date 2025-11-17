"""
Generate all plots for Ministral 8B, Llama 3.1 8B, and Gemma 3 4B models.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from plot_utils import (
    prepare_display_dataframe,
    plot_performance_comparison,
    plot_average_performance,
    prepare_budget_scaling_dataframe,
    plot_average_score_vs_budget,
    plot_benchmark_subplots_vs_budget,
    format_budget_table
)

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'

SELECTED_MODELS = [
    "mistralai/Ministral-8B-Instruct-2410",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-4b-it"
]

def generate_effectiveness_plots():
    """Generate plots for effectiveness on reasoning models analysis."""
    print("=" * 80)
    print("EFFECTIVENESS ON REASONING MODELS ANALYSIS")
    print("=" * 80)
    
    results_df = pd.read_csv('../data/results.csv')
    results_df = results_df[results_df['Model'].isin(SELECTED_MODELS)]
    results_df = results_df[results_df['Prompting'].isin(["Zero-shot", "CoT", "CoT+BF"])]
    results_df = results_df[(results_df['Budget'] == 8192) | (results_df['Budget'].isnull())]
    results_df = results_df[(results_df['Keyword'] == "Wait") | (results_df['Keyword'].isnull())]
    
    print(f"\nFiltered data shape: {results_df.shape}")
    print(f"Models: {results_df['Model'].unique()}")
    
    display_df = prepare_display_dataframe(results_df, apply_styling=False)
    print("\nDisplay DataFrame:")
    print(display_df.to_string())
    
    print("\n1. Generating performance comparison plot...")
    plot_df = plot_performance_comparison(
        display_df, 
        output_path='outputs/fig_new_models_performance_comparison.png'
    )
    
    print("2. Generating average performance plot...")
    plot_average_performance(
        plot_df, 
        output_path='outputs/fig_new_models_average_performance.png'
    )
    
    print("✓ Effectiveness plots generated successfully!")


def generate_scaling_plots():
    """Generate plots for linear scaling analysis."""
    print("\n" + "=" * 80)
    print("LINEAR SCALING ANALYSIS")
    print("=" * 80)
    
    results_df = pd.read_csv('../data/results.csv')
    results_df = results_df[results_df['Model'].isin(SELECTED_MODELS)]
    results_df = results_df[results_df['Prompting'] == "CoT+BF"]
    results_df = results_df[results_df['Keyword'] == "Wait"]
    
    print(f"\nFiltered data shape: {results_df.shape}")
    print(f"Models: {results_df['Model'].unique()}")
    
    table_markdown = format_budget_table(results_df)
    print("\nBudget Scaling Table:")
    print(table_markdown)
    
    results_df = prepare_budget_scaling_dataframe(results_df)
    
    print("\n1. Generating average score vs budget plot...")
    plot_average_score_vs_budget(
        results_df, 
        output_path='outputs/fig_new_models_avg_score.png'
    )
    
    print("2. Generating benchmark subplots vs budget...")
    plot_benchmark_subplots_vs_budget(
        results_df, 
        output_path='outputs/fig_new_models_benchmarks.png'
    )
    
    print("✓ Scaling plots generated successfully!")


if __name__ == "__main__":
    print("Generating plots for:")
    for model in SELECTED_MODELS:
        print(f"  - {model}")
    print()
    
    generate_effectiveness_plots()
    generate_scaling_plots()
    
    print("\n" + "=" * 80)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("=" * 80)
    print("\nOutput files:")
    print("  - outputs/fig_new_models_performance_comparison.png")
    print("  - outputs/fig_new_models_average_performance.png")
    print("  - outputs/fig_new_models_avg_score.png")
    print("  - outputs/fig_new_models_benchmarks.png")


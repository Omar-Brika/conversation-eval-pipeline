import argparse
import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple


def load_reports(file_paths: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parses JSON benchmark reports into Pandas DataFrames."""
    summary_data = []
    detailed_data = []

    for fp in file_paths:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Skipping malformed JSON file -> {fp}")
            continue
        except Exception as e:
            print(f"Warning: Could not read file {fp} -> {e}")
            continue

        model_name = data.get("model", Path(fp).stem)

        # 1. Extract Root level fields
        summary_data.append(
            {
                "Model": model_name,
                "Accuracy (%)": data.get("accuracy", 0),
                "Turn Match Rate (%)": data.get("turn_match_rate", 0),
                "Avg Expected Turns": data.get("avg_expected_turns", 0),
                "Avg Actual Turns": data.get("avg_actual_turns", 0),
                # Totals
                "Total LLM Time (s)": data.get("total_llm_time", 0),
                "Initial Prompter (s)": data.get("total_initial_prompter_time", 0),
                "Routing (s)": data.get("total_routing_voice_cmd_time", 0),
                "Judge (s)": data.get("total_judgment_call_time", 0),
                # Averages
                "Avg Total (s)": data.get("avg_total_llm_time", 0),
                "Avg Initial (s)": data.get("avg_initial_prompter_time", 0),
                "Avg Routing (s)": data.get("avg_routing_voice_cmd_time", 0),
                "Avg Judge (s)": data.get("avg_judgment_call_time", 0),
            }
        )

        # 2. Extract Results level fields
        for res in data.get("results", []):
            detailed_data.append(
                {
                    "Model": model_name,
                    "Domain": res.get("domain", "Unknown"),
                    "Correct": 1 if res.get("correct") else 0,
                    "Test Name": res.get("name", ""),
                }
            )

    df_summary = pd.DataFrame(summary_data)
    df_detailed = pd.DataFrame(detailed_data)

    # Sort summary by Accuracy
    if not df_summary.empty:
        df_summary = df_summary.sort_values(by="Accuracy (%)", ascending=False).reset_index(drop=True)

    return df_summary, df_detailed


def setup_style():
    """Matplotlib and Seaborn Config."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.titleweight"] = "bold"


def plot_overall_performance(df_summary: pd.DataFrame, output_dir: str):
    """Chart 1: Overall Accuracy vs. Turn Match Rate"""
    plt.figure(figsize=(10, 6))

    metrics = ["Accuracy (%)", "Turn Match Rate (%)"]
    df_melted = df_summary.melt(
        id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score"
    )

    ax = sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric")

    plt.title("Overall Model Performance: Accuracy vs. Turn Match Rate")
    plt.ylabel("Score (%)")
    plt.xlabel("Model")
    plt.ylim(0, 105)
    plt.xticks(rotation=15, ha="right")
    plt.legend(title="Metric")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_overall_performance.png"), dpi=300)
    plt.close()


def plot_latency_breakdown(df_summary: pd.DataFrame, output_dir: str):
    """Chart 2: Total LLM Latency Breakdown by Component"""
    plt.figure(figsize=(10, 6))

    time_cols = ["Initial Prompter (s)", "Routing (s)", "Judge (s)"]
    colors = sns.color_palette("muted", len(time_cols))

    bottom = [0] * len(df_summary)

    for i, col in enumerate(time_cols):
        plt.bar(
            df_summary["Model"],
            df_summary[col],
            bottom=bottom,
            label=col,
            color=colors[i],
        )
        bottom = [b + v for b, v in zip(bottom, df_summary[col])]

    plt.title("Total LLM Latency Breakdown by Component")
    plt.ylabel("Total Time (seconds)")
    plt.xlabel("Model")
    plt.xticks(rotation=15, ha="right")
    plt.legend(title="Components", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "02_latency_breakdown.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_domain_accuracy(df_detailed: pd.DataFrame, output_dir: str):
    """Chart 3: Accuracy broken down by conversational domain"""
    if df_detailed.empty:
        return

    domain_acc = (
        df_detailed.groupby(["Model", "Domain"])["Correct"].mean().reset_index()
    )
    domain_acc["Accuracy (%)"] = domain_acc["Correct"] * 100

    top_domains = df_detailed["Domain"].value_counts().nlargest(8).index
    domain_acc_filtered = domain_acc[domain_acc["Domain"].isin(top_domains)]

    plt.figure(figsize=(12, 7))
    sns.barplot(data=domain_acc_filtered, x="Domain", y="Accuracy (%)", hue="Model")

    plt.title("Model Accuracy by Domain")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Domain")
    plt.ylim(0, 105)
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Model")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_domain_accuracy.png"), dpi=300)
    plt.close()


def plot_avg_latency(df_summary: pd.DataFrame, output_dir: str):
    """Chart 4: Average Time per LLM Call by Component"""
    plt.figure(figsize=(10, 6))

    time_cols = ["Avg Initial (s)", "Avg Routing (s)", "Avg Judge (s)"]

    df_melted = df_summary.melt(
        id_vars="Model",
        value_vars=time_cols,
        var_name="Component",
        value_name="Avg Time (s)",
    )

    sns.barplot(data=df_melted, x="Model", y="Avg Time (s)", hue="Component")

    plt.title("Average Latency per LLM Call by Component")
    plt.ylabel("Average Time per Call (seconds)")
    plt.xlabel("Model")
    plt.xticks(rotation=15, ha="right")
    plt.legend(title="Component")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_avg_call_latency.png"), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparative charts from benchmark JSON reports."
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to benchmark JSON report files or glob pattern (e.g., 'reports/*.json')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="charts",
        help="Directory to save the generated charts (default: 'charts')",
    )

    args = parser.parse_args()

    # Resolve glob patterns
    file_paths = []
    for pattern in args.input:
        matched = glob.glob(pattern)
        if not matched and os.path.exists(pattern):
            matched = [pattern]

        # Ensure we only add actual files, not directories
        for path in matched:
            if os.path.isfile(path):
                file_paths.append(path)

    # Deduplicate while preserving order
    file_paths = list(dict.fromkeys(file_paths))

    if not file_paths:
        print("❌ Error: No valid JSON files found matching the input patterns.")
        return

    print(f"📂 Found {len(file_paths)} report file(s). Loading data...")

    df_summary, df_detailed = load_reports(file_paths)

    if df_summary.empty:
        print("❌ Error: No valid data found in the provided JSON files.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    setup_style()

    print("📊 Generating charts...")
    plot_overall_performance(df_summary, args.output_dir)
    plot_latency_breakdown(df_summary, args.output_dir)
    plot_domain_accuracy(df_detailed, args.output_dir)
    plot_avg_latency(df_summary, args.output_dir)

    print(f"✅ Success! {len(file_paths)} models processed.")
    print(f"📁 Charts saved to: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()

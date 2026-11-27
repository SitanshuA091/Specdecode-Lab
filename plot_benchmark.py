
import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = {
    "prompt",
    "method",
    "draft_length",
    "run",
    "generated_tokens",
    "latency_seconds",
    "tokens_per_second",
    "drafted_tokens",
    "accepted_tokens",
    "acceptance_rate_pct",
    "output_matches_baseline",
}


def load_data(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    # Convert numeric columns safely.
    numeric_columns = [
        "draft_length",
        "run",
        "generated_tokens",
        "latency_seconds",
        "tokens_per_second",
        "drafted_tokens",
        "accepted_tokens",
        "acceptance_rate_pct",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["method"] = df["method"].str.lower().str.strip()
    df["output_matches_baseline"] = (
        df["output_matches_baseline"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )

    return df


def get_summary(df):
    summary = (
        df.groupby(["prompt", "method", "draft_length"], dropna=False)
        .agg(
            runs=("run", "count"),
            avg_latency=("latency_seconds", "mean"),
            std_latency=("latency_seconds", "std"),
            avg_tps=("tokens_per_second", "mean"),
            std_tps=("tokens_per_second", "std"),
            avg_acceptance=("acceptance_rate_pct", "mean"),
            avg_drafted=("drafted_tokens", "mean"),
            avg_accepted=("accepted_tokens", "mean"),
            correctness=("output_matches_baseline", "mean"),
        )
        .reset_index()
    )

    return summary


def plot_prompt(prompt, prompt_df, output_dir):
    baseline = prompt_df[prompt_df["method"] == "baseline"]
    speculative = prompt_df[prompt_df["method"] == "speculative"]

    if baseline.empty or speculative.empty:
        print(f"Skipping '{prompt}': baseline or speculative data missing.")
        return

    baseline_latency = baseline["latency_seconds"].mean()
    baseline_tps = baseline["tokens_per_second"].mean()

    spec_summary = (
        speculative.groupby("draft_length")
        .agg(
            latency=("latency_seconds", "mean"),
            latency_std=("latency_seconds", "std"),
            tps=("tokens_per_second", "mean"),
            tps_std=("tokens_per_second", "std"),
            acceptance=("acceptance_rate_pct", "mean"),
            drafted=("drafted_tokens", "mean"),
            accepted=("accepted_tokens", "mean"),
        )
        .sort_index()
    )

    k_values = spec_summary.index.astype(int).astype(str).tolist()
    x = range(len(k_values))

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(f"Speculative Decoding Benchmark\nPrompt: {prompt}", fontsize=14)

    # 1. Latency comparison
    ax = axes[0, 0]
    latency_values = spec_summary["latency"].values
    latency_errors = spec_summary["latency_std"].fillna(0).values

    ax.bar(k_values, latency_values, yerr=latency_errors, capsize=4,
           label="Speculative")
    ax.axhline(
        baseline_latency,
        linestyle="--",
        label=f"Baseline ({baseline_latency:.2f}s)",
    )
    ax.set_title("Average Latency")
    ax.set_xlabel("Draft length (K)")
    ax.set_ylabel("Seconds")
    ax.legend()

    # 2. Throughput comparison
    ax = axes[0, 1]
    tps_values = spec_summary["tps"].values
    tps_errors = spec_summary["tps_std"].fillna(0).values

    ax.bar(k_values, tps_values, yerr=tps_errors, capsize=4,
           label="Speculative")
    ax.axhline(
        baseline_tps,
        linestyle="--",
        label=f"Baseline ({baseline_tps:.2f} tok/s)",
    )
    ax.set_title("Generation Throughput")
    ax.set_xlabel("Draft length (K)")
    ax.set_ylabel("Tokens / second")
    ax.legend()

    # 3. Acceptance rate
    ax = axes[0, 2]
    ax.plot(
        k_values,
        spec_summary["acceptance"],
        marker="o",
    )
    ax.set_title("Draft Acceptance Rate")
    ax.set_xlabel("Draft length (K)")
    ax.set_ylabel("Accepted proposals (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # 4. Speedup relative to baseline
    ax = axes[1, 0]
    speedup = baseline_latency / spec_summary["latency"]

    ax.bar(k_values, speedup)
    ax.axhline(1.0, linestyle="--", label="Baseline parity (1x)")
    ax.set_title("Speedup vs Baseline")
    ax.set_xlabel("Draft length (K)")
    ax.set_ylabel("Speedup (x)")
    ax.legend()

    # 5. Drafted vs accepted tokens
    ax = axes[1, 1]
    width = 0.35

    ax.bar(
        [i - width / 2 for i in x],
        spec_summary["drafted"],
        width,
        label="Drafted",
    )
    ax.bar(
        [i + width / 2 for i in x],
        spec_summary["accepted"],
        width,
        label="Accepted",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(k_values)
    ax.set_title("Drafted vs Accepted Tokens")
    ax.set_xlabel("Draft length (K)")
    ax.set_ylabel("Average tokens")
    ax.legend()

    # 6. Correctness summary
    ax = axes[1, 2]
    ax.axis("off")

    total_runs = len(prompt_df)
    matching_runs = int(prompt_df["output_matches_baseline"].sum())
    all_match = matching_runs == total_runs

    ax.text(
        0.05,
        0.85,
        "Output correctness",
        fontsize=13,
        weight="bold",
    )
    ax.text(
        0.05,
        0.65,
        f"Matching runs: {matching_runs}/{total_runs}",
        fontsize=12,
    )
    ax.text(
        0.05,
        0.48,
        f"Baseline avg latency: {baseline_latency:.2f}s",
        fontsize=11,
    )
    ax.text(
        0.05,
        0.33,
        f"Best speculative K: {spec_summary['latency'].idxmin()}",
        fontsize=11,
    )
    ax.text(
        0.05,
        0.18,
        "All outputs match baseline" if all_match
        else "Some outputs differ from baseline",
        fontsize=11,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    # Create a filesystem-safe filename.
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", prompt).strip("_").lower()
    output_path = output_dir / f"benchmark_{slug}.png"

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {output_path.resolve()}")

    return fig


def print_summary(summary):
    print("\n========== BENCHMARK SUMMARY ==========")

    for prompt in summary["prompt"].unique():
        print(f"\nPrompt: {prompt}")

        prompt_summary = summary[summary["prompt"] == prompt]

        for _, row in prompt_summary.iterrows():
            if row["method"] == "baseline":
                method_name = "Baseline"
            else:
                method_name = f"Speculative K={int(row['draft_length'])}"

            print(
                f"{method_name:<18} | "
                f"Latency: {row['avg_latency']:.2f}s | "
                f"Throughput: {row['avg_tps']:.3f} tok/s"
            )

            if row["method"] == "speculative":
                print(
                    f"{'':18} | "
                    f"Acceptance: {row['avg_acceptance']:.2f}% | "
                    f"Accepted: {row['avg_accepted']:.1f}/"
                    f"{row['avg_drafted']:.1f}"
                )

    print("\n=======================================")


def main():
    parser = argparse.ArgumentParser(
        description="Plot speculative decoding benchmark results."
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmark_history.csv",
        help="Path to benchmark_history.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmark_plots",
        help="Directory to save generated plots",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively",
    )

    args = parser.parse_args()

    df = load_data(args.csv)
    summary = get_summary(df)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(df)} benchmark records from {args.csv}")
    print_summary(summary)

    figures = []

    for prompt in df["prompt"].dropna().unique():
        prompt_df = df[df["prompt"] == prompt]
        fig = plot_prompt(prompt, prompt_df, args.output_dir)

        if fig is not None:
            figures.append(fig)

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()

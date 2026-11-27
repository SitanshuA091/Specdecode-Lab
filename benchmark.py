
import argparse
import csv
import time
from pathlib import Path
from datetime import datetime

import torch

from speculative import (
    DEVICE, load_models, prepare_prompt, speculative_decode,
)


# Timing utilities

def synchronize_device():
    """Ensure GPU operations finish before recording time."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def start_timer():
    synchronize_device()
    return time.perf_counter()


def stop_timer(start_time):
    synchronize_device()
    return time.perf_counter() - start_time

# Target-only greedy baseline

@torch.inference_mode()
def run_baseline(tokenizer, target_model, prompt, max_new_tokens):
    """
    Standard autoregressive greedy decoding.

    Uses the target model's KV cache between generated tokens.
    """

    input_ids, attention_mask = prepare_prompt(tokenizer, prompt)

    outputs = target_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )

    past_key_values = outputs.past_key_values

    next_token = outputs.logits[:, -1, :].argmax(
        dim=-1,
        keepdim=True,
    )

    generated_tokens = []

    eos_token_id = tokenizer.eos_token_id

    for step in range(max_new_tokens):

        generated_tokens.append(next_token)

        if (
            eos_token_id is not None
            and next_token.item() == eos_token_id
        ):
            break

        if step == max_new_tokens - 1:
            break

        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (1, 1),
                    dtype=attention_mask.dtype,
                    device=DEVICE,
                ),
            ],
            dim=-1,
        )

        outputs = target_model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        past_key_values = outputs.past_key_values

        next_token = outputs.logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

    if generated_tokens:
        return torch.cat(generated_tokens, dim=-1)

    return torch.empty(
        (1, 0),
        dtype=torch.long,
        device=DEVICE,
    )

# Metrics
def calculate_metrics(generated_ids, elapsed):
    token_count = generated_ids.shape[-1]

    tokens_per_second = (
        token_count / elapsed
        if elapsed > 0
        else 0.0
    )

    return {
        "generated_tokens": token_count,
        "latency_seconds": elapsed,
        "tokens_per_second": tokens_per_second,
    }

# Benchmark execution
@torch.inference_mode()
def run_benchmark(
    tokenizer,
    target_model,
    draft_model,
    prompt,
    draft_lengths,
    max_new_tokens,
    repeats,
    warmup,
):
    results = []
    
    # Baseline warm-up
    print("\nRunning baseline warm-up...")

    for _ in range(warmup):
        run_baseline(
            tokenizer,
            target_model,
            prompt,
            max_new_tokens,
        )

    # Baseline measurements

    baseline_runs = []
    print("\nBenchmarking baseline...")

    for run_number in range(1, repeats + 1):
        start = start_timer()

        baseline_ids = run_baseline(
            tokenizer,
            target_model,
            prompt,
            max_new_tokens,
        )

        elapsed = stop_timer(start)
        metrics = calculate_metrics(baseline_ids, elapsed)

        baseline_runs.append({
            "run": run_number,
            "ids": baseline_ids.clone(),
            "metrics": metrics,
        })

        results.append({
            "method": "baseline",
            "draft_length": "",
            "run": run_number,
            **metrics,
            "drafted_tokens": "",
            "accepted_tokens": "",
            "acceptance_rate_pct": "",
            "output_matches_baseline": True,
        })

        print(
            f"Baseline run {run_number}: "
            f"{metrics['tokens_per_second']:.3f} tokens/sec"
        )

    # Use the first baseline run as the reference output
    reference_ids = baseline_runs[0]["ids"]

    baseline_average_tps = sum(
        run["metrics"]["tokens_per_second"]
        for run in baseline_runs
    ) / repeats

    # Speculative measurements

    for k in draft_lengths:
        print(f"\nBenchmarking speculative decoding: K={k}")

        # Warm-up for this draft length.
        for _ in range(warmup):
            speculative_decode(
                tokenizer=tokenizer,
                target_model=target_model,
                draft_model=draft_model,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                draft_length=k,
            )

        for run_number in range(1, repeats + 1):
            start = start_timer()
            (
                generated_text,
                generated_ids,
                drafted_tokens,
                accepted_tokens,
                acceptance_rate,
                internal_elapsed,
            ) = speculative_decode(
                tokenizer=tokenizer,
                target_model=target_model,
                draft_model=draft_model,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                draft_length=k,
            )

            elapsed = stop_timer(start)
            metrics = calculate_metrics(
                generated_ids,
                elapsed,
            )

            output_matches = torch.equal(
                generated_ids,
                reference_ids,
            )

            results.append({
                "method": "speculative",
                "draft_length": k,
                "run": run_number,
                **metrics,
                "drafted_tokens": drafted_tokens,
                "accepted_tokens": accepted_tokens,
                "acceptance_rate_pct": acceptance_rate,
                "output_matches_baseline": output_matches,
            })

            print(
                f"K={k}, run={run_number}: "
                f"{metrics['tokens_per_second']:.3f} tokens/sec | "
                f"Acceptance={acceptance_rate:.2f}% | "
                f"Output match={output_matches}"
            )

    return results, baseline_average_tps

# CSV export and summary

def save_results(results, output_file, prompt):

    if not results:
        print("No benchmark results to save.")
        return

    output_path = Path(output_file)

    # Add experiment metadata to every row.
    timestamp = datetime.now().isoformat(timespec="seconds")

    enriched_results = []

    for result in results:
        enriched_results.append({
            "timestamp": timestamp,
            "prompt": prompt,
            **result,
        })

    fieldnames = list(enriched_results[0].keys())

    # Append instead of overwriting previous experiments.
    file_exists = output_path.exists() and output_path.stat().st_size > 0

    with open(
        output_path,
        mode="a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(enriched_results)

    print(f"\nResults appended to: {output_path.resolve()}")
    print(f"Records added: {len(enriched_results)}")



def print_summary(results, baseline_average_tps):

    print("\n" + "=" * 65)
    print("BENCHMARK SUMMARY")
    print("=" * 65)

    print(
        f"\nBaseline average throughput: "
        f"{baseline_average_tps:.3f} tokens/sec"
    )

    print(
        f"\n{'Method':<15}"
        f"{'K':<8}"
        f"{'Latency':<14}"
        f"{'Tokens/sec':<14}"
        f"{'Acceptance':<14}"
        f"{'Speedup':<10}"
    )

    print("-" * 75)

    grouped = {}

    for result in results:

        key = (
            result["method"],
            result["draft_length"],
        )

        grouped.setdefault(key, []).append(result)

    for (method, k), runs in grouped.items():

        average_latency = sum(
            row["latency_seconds"] for row in runs
        ) / len(runs)

        average_tps = sum(
            row["tokens_per_second"] for row in runs
        ) / len(runs)

        if method == "baseline":
            acceptance = "-"
            speedup = 1.0
        else:
            acceptance = (
                sum(
                    row["acceptance_rate_pct"]
                    for row in runs
                ) / len(runs)
            )

            speedup = (
                average_tps / baseline_average_tps
                if baseline_average_tps > 0
                else 0.0
            )

            acceptance = f"{acceptance:.2f}%"

        print(
            f"{method:<15}"
            f"{str(k):<8}"
            f"{average_latency:<14.2f}"
            f"{average_tps:<14.3f}"
            f"{acceptance:<14}"
            f"{speedup:.3f}x"
        )

    print("\nSpeedup = speculative throughput / baseline throughput")

# Main
def main():

    parser = argparse.ArgumentParser(
        description="Benchmark greedy and speculative decoding."
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--draft-lengths",
        type=int,
        nargs="+",
        default=[3, 5, 8],
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_history.csv",
    )

    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    if args.warmup < 0:
        parser.error("--warmup cannot be negative")

    if any(k < 1 for k in args.draft_lengths):
        parser.error("All draft lengths must be at least 1")

    # Load models only once.
    tokenizer, target_model, draft_model = load_models()

    results, baseline_average_tps = run_benchmark(
        tokenizer=tokenizer,
        target_model=target_model,
        draft_model=draft_model,
        prompt=args.prompt,
        draft_lengths=args.draft_lengths,
        max_new_tokens=args.max_new_tokens,
        repeats=args.repeats,
        warmup=args.warmup,
    )

    save_results(results=results,
                 output_file=args.output,
                 prompt=args.prompt,
                 )

    print_summary(
        results,
        baseline_average_tps,
    )


if __name__ == "__main__":
    main()

"""
Lab 7: LLM Energy and Latency Profiling with MLC-LLM
=====================================================
This script measures energy consumption and latency for LLM inference
across different prompt token (PT) and generation token (GT) combinations.
"""

import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Token length configurations
PROMPT_TOKENS = [64, 256, 512, 1024]  # PT values
GENERATION_TOKENS = [16, 64, 128]  # GT values
NUM_TRIALS = 3  # Runs per configuration

# Output files
CSV_OUTPUT = "energy_results.csv"
PLOT_OUTPUT_DIR = "plots"
PROMPTS_FILE = "prompts.json"  # External prompts config

# MLC-LLM model path (adjust based on your setup)
MODEL_PATH = "HF://mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC"  # Example model


# -----------------------------------------------------------------------------
# Prompt Loading
# -----------------------------------------------------------------------------


def save_prompts(prompts: Dict[int, str], topic: str, filename: str = PROMPTS_FILE):
    """Save prompts to JSON file for reproducibility."""
    data = {
        "topic": topic,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompts": {str(k): v for k, v in prompts.items()},
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Prompts saved to {filename}")


def load_prompts(filename: str = PROMPTS_FILE) -> Dict[int, str]:
    """Load prompts from JSON file."""
    with open(filename, "r") as f:
        data = json.load(f)
    print(
        f"Loaded prompts for topic: {data['topic']} (generated: {data['generated_at']})"
    )
    return {int(k): v for k, v in data["prompts"].items()}


def get_prompts() -> Dict[int, str]:
    """Load prompts from prompts.json file."""
    if not os.path.exists(PROMPTS_FILE):
        raise FileNotFoundError(
            f"{PROMPTS_FILE} not found. Please create a prompts.json file with your prompts."
        )
    return load_prompts(PROMPTS_FILE)


# Prompts will be loaded lazily to avoid issues at import time
PROMPTS = None


def ensure_prompts_loaded():
    """Ensure prompts are loaded (lazy loading)."""
    global PROMPTS
    if PROMPTS is None:
        PROMPTS = get_prompts()


# -----------------------------------------------------------------------------
# Energy Measurement Utilities
# -----------------------------------------------------------------------------


@dataclass
class MeasurementResult:
    """Stores results from a single inference run."""

    prompt_tokens: int
    generation_tokens: int
    energy_mj: float
    latency_s: float
    trial: int


def get_energy_macos(duration_s: float) -> float:
    """
    Estimate energy on macOS using powermetrics (requires sudo).
    Returns energy in millijoules (mJ).

    Alternative: If you don't have sudo access, this returns an estimate
    based on typical CPU power draw.
    """
    # Option 1: Use powermetrics (requires sudo)
    # Uncomment and modify if you have sudo access:
    # try:
    #     result = subprocess.run(
    #         ['sudo', 'powermetrics', '-i', '100', '-n', '1', '--samplers', 'cpu_power'],
    #         capture_output=True, text=True, timeout=5
    #     )
    #     # Parse power from output and multiply by duration
    #     # power_w = parse_power(result.stdout)
    #     # return power_w * duration_s * 1000  # Convert to mJ
    # except:
    #     pass

    # Option 2: Estimate based on typical MacBook CPU power
    # M1/M2 chips typically draw 10-30W under load
    # This is a rough estimate - replace with actual measurements if possible
    estimated_power_w = 15.0  # Watts (adjust based on your Mac)
    return estimated_power_w * duration_s * 1000  # mJ


def get_energy_android_perfetto(trace_file: str) -> float:
    """
    Parse energy from a Perfetto trace file (for Android devices).
    Returns energy in millijoules (mJ).

    You'll need to run Perfetto separately and provide the trace file.
    """
    # TODO: Implement Perfetto trace parsing
    # This would involve parsing the protobuf trace file
    raise NotImplementedError("Implement Perfetto parsing for Android")


# -----------------------------------------------------------------------------
# MLC-LLM Inference
# -----------------------------------------------------------------------------


class MLCInference:
    """
    Wrapper for MLC-LLM inference. Keeps engine loaded between calls.
    """

    _instance = None
    _engine = None

    @classmethod
    def get_engine(cls):
        """Get or create the MLC engine (singleton pattern)."""
        if cls._engine is None:
            from mlc_llm import MLCEngine

            print(f"Loading MLC-LLM model: {MODEL_PATH}")
            cls._engine = MLCEngine(MODEL_PATH)
            print("Model loaded successfully!")
        return cls._engine

    @classmethod
    def cleanup(cls):
        """Cleanup the engine when done."""
        if cls._engine is not None:
            cls._engine.terminate()
            cls._engine = None


def run_inference_mlc(prompt: str, max_new_tokens: int) -> Tuple[float, str]:
    """
    Run inference using MLC-LLM and return (latency_seconds, output_text).
    """
    engine = MLCInference.get_engine()

    start_time = time.perf_counter()

    response = engine.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
    )

    end_time = time.perf_counter()
    latency = end_time - start_time

    output_text = response.choices[0].message.content

    return latency, output_text


# -----------------------------------------------------------------------------
# Experiment Runner
# -----------------------------------------------------------------------------


def run_experiment() -> List[MeasurementResult]:
    """
    Run the full experiment across all PT x GT combinations.

    Returns:
        List of MeasurementResult objects.
    """
    # Ensure prompts are loaded
    ensure_prompts_loaded()

    results = []
    total_runs = len(PROMPT_TOKENS) * len(GENERATION_TOKENS) * NUM_TRIALS
    current_run = 0

    print(f"Starting experiment: {total_runs} total runs")
    print(f"Model: {MODEL_PATH}")
    print(f"PT values: {PROMPT_TOKENS}")
    print(f"GT values: {GENERATION_TOKENS}")
    print(f"Trials per configuration: {NUM_TRIALS}")
    print("-" * 60)

    for pt in PROMPT_TOKENS:
        prompt = PROMPTS[pt]

        for gt in GENERATION_TOKENS:
            for trial in range(1, NUM_TRIALS + 1):
                current_run += 1
                print(
                    f"[{current_run}/{total_runs}] PT={pt}, GT={gt}, Trial={trial}...",
                    end=" ",
                )

                # Run inference and measure
                latency, _ = run_inference_mlc(prompt, gt)
                energy = get_energy_macos(latency)

                result = MeasurementResult(
                    prompt_tokens=pt,
                    generation_tokens=gt,
                    energy_mj=energy,
                    latency_s=latency,
                    trial=trial,
                )
                results.append(result)

                print(f"Latency={latency:.3f}s, Energy={energy:.1f}mJ")

    print("-" * 60)
    print(f"Experiment complete. {len(results)} measurements collected.")

    return results


def save_results_csv(results: List[MeasurementResult], filename: str):
    """Save results to CSV file."""
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["PT", "GT", "Energy(mJ)", "Latency(s)", "Trial"])

        for r in results:
            writer.writerow(
                [
                    r.prompt_tokens,
                    r.generation_tokens,
                    f"{r.energy_mj:.2f}",
                    f"{r.latency_s:.4f}",
                    r.trial,
                ]
            )

    print(f"Results saved to {filename}")


def load_results_csv(filename: str) -> List[MeasurementResult]:
    """Load results from CSV file."""
    results = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(
                MeasurementResult(
                    prompt_tokens=int(row["PT"]),
                    generation_tokens=int(row["GT"]),
                    energy_mj=float(row["Energy(mJ)"]),
                    latency_s=float(row["Latency(s)"]),
                    trial=int(row["Trial"]),
                )
            )
    return results


# -----------------------------------------------------------------------------
# Energy Predictor Model
# -----------------------------------------------------------------------------


def fit_energy_predictor(
    results: List[MeasurementResult],
) -> Tuple[float, float, float]:
    """
    Fit a linear energy predictor: E = α × PT + β × GT + γ

    Returns:
        (alpha, beta, gamma) coefficients
    """
    # Prepare data
    X = np.array([[r.prompt_tokens, r.generation_tokens, 1] for r in results])
    y = np.array([r.energy_mj for r in results])

    # Least squares fit: X @ coeffs = y
    # coeffs = (X^T X)^-1 X^T y
    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    alpha, beta, gamma = coeffs

    # Calculate R-squared
    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f"\nEnergy Predictor: E = {alpha:.4f} × PT + {beta:.4f} × GT + {gamma:.4f}")
    print(f"R-squared: {r_squared:.4f}")

    return alpha, beta, gamma


def predict_energy(pt: int, gt: int, alpha: float, beta: float, gamma: float) -> float:
    """Predict energy using the fitted model."""
    return alpha * pt + beta * gt + gamma


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------


def create_visualizations(
    results: List[MeasurementResult], alpha: float, beta: float, gamma: float
):
    """Create all required plots."""
    import matplotlib.pyplot as plt

    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

    # Aggregate results by (PT, GT) - average across trials
    from collections import defaultdict

    aggregated = defaultdict(list)
    for r in results:
        aggregated[(r.prompt_tokens, r.generation_tokens)].append(r.energy_mj)

    avg_energy = {k: np.mean(v) for k, v in aggregated.items()}

    # Plot 1: Energy vs PT for fixed GT
    plt.figure(figsize=(10, 6))
    for gt in GENERATION_TOKENS:
        pts = sorted([k[0] for k in avg_energy.keys() if k[1] == gt])
        energies = [avg_energy[(pt, gt)] for pt in pts]
        plt.plot(pts, energies, "o-", label=f"GT={gt}", linewidth=2, markersize=8)

    plt.xlabel("Prompt Tokens (PT)", fontsize=12)
    plt.ylabel("Energy (mJ)", fontsize=12)
    plt.title("Energy vs Prompt Tokens for Different Generation Lengths", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{PLOT_OUTPUT_DIR}/energy_vs_pt.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOT_OUTPUT_DIR}/energy_vs_pt.png")

    # Plot 2: Energy vs GT for fixed PT
    plt.figure(figsize=(10, 6))
    for pt in PROMPT_TOKENS:
        gts = sorted([k[1] for k in avg_energy.keys() if k[0] == pt])
        energies = [avg_energy[(pt, gt)] for gt in gts]
        plt.plot(gts, energies, "o-", label=f"PT={pt}", linewidth=2, markersize=8)

    plt.xlabel("Generation Tokens (GT)", fontsize=12)
    plt.ylabel("Energy (mJ)", fontsize=12)
    plt.title("Energy vs Generation Tokens for Different Prompt Lengths", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{PLOT_OUTPUT_DIR}/energy_vs_gt.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOT_OUTPUT_DIR}/energy_vs_gt.png")

    # Plot 3: Predicted vs Measured Energy
    plt.figure(figsize=(8, 8))
    measured = [r.energy_mj for r in results]
    predicted = [
        predict_energy(r.prompt_tokens, r.generation_tokens, alpha, beta, gamma)
        for r in results
    ]

    plt.scatter(measured, predicted, alpha=0.6, s=50)

    # Add perfect prediction line
    min_val = min(min(measured), min(predicted))
    max_val = max(max(measured), max(predicted))
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Perfect prediction",
    )

    plt.xlabel("Measured Energy (mJ)", fontsize=12)
    plt.ylabel("Predicted Energy (mJ)", fontsize=12)
    plt.title("Predicted vs Measured Energy", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.savefig(
        f"{PLOT_OUTPUT_DIR}/predicted_vs_measured.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"Saved: {PLOT_OUTPUT_DIR}/predicted_vs_measured.png")

    # Bonus: Latency plots
    # Aggregate latency
    latency_agg = defaultdict(list)
    for r in results:
        latency_agg[(r.prompt_tokens, r.generation_tokens)].append(r.latency_s)
    avg_latency = {k: np.mean(v) for k, v in latency_agg.items()}

    plt.figure(figsize=(10, 6))
    for gt in GENERATION_TOKENS:
        pts = sorted([k[0] for k in avg_latency.keys() if k[1] == gt])
        latencies = [avg_latency[(pt, gt)] for pt in pts]
        plt.plot(pts, latencies, "o-", label=f"GT={gt}", linewidth=2, markersize=8)

    plt.xlabel("Prompt Tokens (PT)", fontsize=12)
    plt.ylabel("Latency (s)", fontsize=12)
    plt.title("Latency vs Prompt Tokens for Different Generation Lengths", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{PLOT_OUTPUT_DIR}/latency_vs_pt.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOT_OUTPUT_DIR}/latency_vs_pt.png")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("Lab 7: LLM Energy and Latency Profiling")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  MODEL_PATH: {MODEL_PATH}")

    # Check if we should load existing results or run new experiment
    if os.path.exists(CSV_OUTPUT):
        response = (
            input(
                f"\n{CSV_OUTPUT} exists. Load existing (L) or run new experiment (N)? "
            )
            .strip()
            .upper()
        )
        if response == "L":
            results = load_results_csv(CSV_OUTPUT)
            print(f"Loaded {len(results)} measurements from {CSV_OUTPUT}")
        else:
            results = run_experiment()
            save_results_csv(results, CSV_OUTPUT)
    else:
        results = run_experiment()
        save_results_csv(results, CSV_OUTPUT)

    # Cleanup MLC engine
    MLCInference.cleanup()

    # Fit energy predictor
    alpha, beta, gamma = fit_energy_predictor(results)

    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(results, alpha, beta, gamma)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    from collections import defaultdict

    by_config = defaultdict(list)
    for r in results:
        by_config[(r.prompt_tokens, r.generation_tokens)].append(
            (r.energy_mj, r.latency_s)
        )

    print(f"{'PT':<6} {'GT':<6} {'Avg Energy (mJ)':<18} {'Avg Latency (s)':<18}")
    print("-" * 50)
    for (pt, gt), measurements in sorted(by_config.items()):
        avg_e = np.mean([m[0] for m in measurements])
        avg_l = np.mean([m[1] for m in measurements])
        print(f"{pt:<6} {gt:<6} {avg_e:<18.2f} {avg_l:<18.4f}")

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print(f"Results: {CSV_OUTPUT}")
    print(f"Plots: {PLOT_OUTPUT_DIR}/")
    print("=" * 60)


def cli():
    """Command-line interface for the experiment."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Lab 7: LLM Energy and Latency Profiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiment
  python energy_experiment.py

  # Just generate plots from existing CSV
  python energy_experiment.py --plot-only

  # Use a different model
  python energy_experiment.py --model "HF://mlc-ai/other-model"
        """,
    )

    parser.add_argument(
        "--plot-only", action="store_true", help="Only generate plots from existing CSV"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="MLC-LLM model path (overrides MODEL_PATH)",
    )

    args = parser.parse_args()

    # Update globals based on args
    global MODEL_PATH

    if args.model:
        MODEL_PATH = args.model

    ensure_prompts_loaded()

    # Plot only mode
    if args.plot_only:
        if not os.path.exists(CSV_OUTPUT):
            print(f"Error: {CSV_OUTPUT} not found. Run experiment first.")
            return
        results = load_results_csv(CSV_OUTPUT)
        alpha, beta, gamma = fit_energy_predictor(results)
        create_visualizations(results, alpha, beta, gamma)
        return

    # Run main experiment
    main()


if __name__ == "__main__":
    cli()

#!/usr/bin/env python3
"""
Validate and Clean STT Evaluation Results

Identifies and removes failed transcriptions (WER=1.0 with empty hypothesis)
that would distort aggregate statistics, then regenerates clean summaries.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import shutil

SCRIPT_DIR = Path(__file__).parent
RESULTS_PATH = SCRIPT_DIR / "results"
INDIVIDUAL_PATH = RESULTS_PATH / "individual"


@dataclass
class ModelSummary:
    """Summary statistics for a model."""
    model: str
    total_files: int
    average_wer: float
    median_wer: float
    min_wer: float
    max_wer: float
    total_time_seconds: float
    avg_time_per_file: float


def find_failures(threshold: float = 1.0) -> tuple[list[dict], set[str]]:
    """
    Find all result files with WER >= threshold and empty/error hypothesis.
    Returns (failures_list, set_of_failed_audio_files).
    """
    failures = []
    failed_audio_files = set()

    if not INDIVIDUAL_PATH.exists():
        print(f"No individual results found at {INDIVIDUAL_PATH}")
        return failures, failed_audio_files

    for model_dir in INDIVIDUAL_PATH.iterdir():
        if not model_dir.is_dir():
            continue

        for result_file in model_dir.glob("*.json"):
            with open(result_file) as f:
                data = json.load(f)

            wer = data.get("wer", 0)
            hypothesis = data.get("hypothesis", "").strip()

            # Identify failures: WER=1.0 AND empty/error hypothesis
            is_failure = (
                wer >= threshold and
                (hypothesis == "" or
                 hypothesis.startswith("[TIMEOUT]") or
                 hypothesis.startswith("[ERROR"))
            )

            if is_failure:
                audio_file = data.get("audio_file")
                failed_audio_files.add(audio_file)
                failures.append({
                    "file": result_file,
                    "model": data.get("model"),
                    "audio_file": audio_file,
                    "wer": wer,
                    "hypothesis": hypothesis[:50] + "..." if len(hypothesis) > 50 else hypothesis,
                    "reference": data.get("reference", "")[:50] + "..."
                })

    return failures, failed_audio_files


def calculate_clean_summary(model_name: str, results: list[dict]) -> ModelSummary:
    """Calculate summary statistics excluding failures."""
    if not results:
        return ModelSummary(
            model=model_name,
            total_files=0,
            average_wer=0,
            median_wer=0,
            min_wer=0,
            max_wer=0,
            total_time_seconds=0,
            avg_time_per_file=0
        )

    wers = [r["wer"] for r in results]
    times = [r["duration_seconds"] for r in results]
    sorted_wers = sorted(wers)

    return ModelSummary(
        model=model_name,
        total_files=len(results),
        average_wer=sum(wers) / len(wers),
        median_wer=sorted_wers[len(sorted_wers) // 2],
        min_wer=min(wers),
        max_wer=max(wers),
        total_time_seconds=sum(times),
        avg_time_per_file=sum(times) / len(times)
    )


def regenerate_clean_results(exclude_audio_files: set[str] = None) -> tuple[dict, list]:
    """
    Regenerate results excluding specified audio files from ALL models.
    This ensures fair comparison by removing problematic samples everywhere.
    Returns (all_results, summaries).
    """
    all_results = {}
    exclude_audio_files = exclude_audio_files or set()

    if not INDIVIDUAL_PATH.exists():
        print(f"No individual results found at {INDIVIDUAL_PATH}")
        return {}, []

    for model_dir in sorted(INDIVIDUAL_PATH.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        results = []
        excluded_count = 0

        for result_file in sorted(model_dir.glob("*.json")):
            with open(result_file) as f:
                data = json.load(f)

            audio_file = data.get("audio_file", "")

            # Exclude this sample across ALL models for fairness
            if audio_file in exclude_audio_files:
                excluded_count += 1
                continue

            results.append(data)

        all_results[model_name] = results
        if excluded_count > 0:
            print(f"  {model_name}: excluded {excluded_count} sample(s)")

    # Calculate summaries
    summaries = []
    for model_name, results in all_results.items():
        summary = calculate_clean_summary(model_name, results)
        summaries.append(summary)

    return all_results, summaries


def print_comparison(original_summaries: list[ModelSummary],
                     cleaned_summaries: list[ModelSummary]):
    """Print comparison between original and cleaned results."""
    print("\n" + "="*80)
    print("ORIGINAL VS CLEANED COMPARISON")
    print("="*80)

    orig_dict = {s.model: s for s in original_summaries}
    clean_dict = {s.model: s for s in cleaned_summaries}

    print(f"\n{'Model':<20} {'Orig Avg WER':>12} {'Clean Avg WER':>14} {'Orig Max':>10} {'Clean Max':>10}")
    print("-"*80)

    for model in sorted(orig_dict.keys()):
        orig = orig_dict.get(model)
        clean = clean_dict.get(model)

        if orig and clean:
            print(f"{model:<20} {orig.average_wer:>11.2%} {clean.average_wer:>13.2%} "
                  f"{orig.max_wer:>9.2%} {clean.max_wer:>9.2%}")


def save_clean_results_with_metadata(all_results: dict, summaries: list[ModelSummary],
                                      excluded_samples: set[str], suffix: str = "cleaned"):
    """Save cleaned results with metadata about excluded samples."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save summary CSV
    csv_path = RESULTS_PATH / f"summary_{suffix}_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        for s in summaries:
            writer.writerow(asdict(s))
    print(f"\nCleaned summary saved to: {csv_path}")

    # Save detailed JSON with exclusion metadata
    json_path = RESULTS_PATH / f"detailed_results_{suffix}_{timestamp}.json"
    json_data = {
        "timestamp": timestamp,
        "note": "Samples with complete failures excluded from ALL models for fair comparison",
        "excluded_samples": sorted(excluded_samples),
        "original_sample_count": len(all_results.get(list(all_results.keys())[0], [])) + len(excluded_samples),
        "cleaned_sample_count": len(all_results.get(list(all_results.keys())[0], [])),
        "summaries": [asdict(s) for s in summaries],
        "results": all_results
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Cleaned detailed results saved to: {json_path}")

    return csv_path, json_path


def main():
    print("="*60)
    print("STT RESULTS VALIDATION & CLEANUP")
    print("="*60)
    print(f"\nScanning: {INDIVIDUAL_PATH}")

    # Find failures
    failures, failed_audio_files = find_failures()

    if not failures:
        print("\nNo failures found - all results are valid!")
        return

    print(f"\nFound {len(failures)} failure(s) in {len(failed_audio_files)} unique sample(s):")
    print("-"*60)
    for f in failures:
        print(f"  Model: {f['model']}")
        print(f"  File:  {f['audio_file']}")
        print(f"  WER:   {f['wer']:.2%}")
        print(f"  Hyp:   '{f['hypothesis']}'")
        print()

    print(f"Samples to exclude from ALL models: {sorted(failed_audio_files)}")

    # Generate original summaries (with all samples)
    print("\nCalculating original summaries (all samples)...")
    orig_results, orig_summaries = regenerate_clean_results(exclude_audio_files=set())

    # Generate cleaned summaries (excluding failed samples from ALL models)
    print("\nCalculating cleaned summaries (excluding failed samples from ALL models)...")
    clean_results, clean_summaries = regenerate_clean_results(exclude_audio_files=failed_audio_files)

    # Print comparison
    print_comparison(orig_summaries, clean_summaries)

    # Save cleaned results
    print("\nSaving cleaned results...")
    save_clean_results_with_metadata(clean_results, clean_summaries, failed_audio_files)

    print("\nDone! Failed samples excluded from ALL models for fair comparison.")


if __name__ == "__main__":
    main()

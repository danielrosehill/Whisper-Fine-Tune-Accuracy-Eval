#!/usr/bin/env python3
"""
STT Fine-Tune Evaluation Script

Compares fine-tuned Whisper models against original OpenAI models.
Uses whisper-normalizer for fair text comparison (handles "3000" vs "three thousand", etc.)
Runs ONE transcription at a time for accurate measurements.
"""

import subprocess
import json
import csv
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import tempfile

import werpy
from whisper_normalizer.english import EnglishTextNormalizer

# Paths
SCRIPT_DIR = Path(__file__).parent
MODELS_CONFIG = SCRIPT_DIR / "models.json"
DATASET_PATH = Path.home() / "repos/hugging-face/datasets/private/Small-STT-Eval-Audio-Dataset/data"
METADATA_CSV = DATASET_PATH / "metadata.csv"
OUTPUT_PATH = SCRIPT_DIR / "results"


def load_models_from_config() -> dict[str, Path]:
    """Load model paths from models.json config file."""
    with open(MODELS_CONFIG) as f:
        config = json.load(f)

    models = {}

    # Load fine-tuned GGML models
    ft_ggml = config["fine_tuned"]["ggml"]
    ft_base = Path(ft_ggml["base_path"])
    for size, filename in ft_ggml["models"].items():
        models[f"finetune-{size}"] = ft_base / filename

    # Load original GGML models
    orig_ggml = config["original"]["ggml"]
    orig_base = Path(orig_ggml["base_path"])
    for size, filename in orig_ggml["models"].items():
        models[f"original-{size}"] = orig_base / filename

    return models


@dataclass
class TranscriptionResult:
    """Single transcription result."""
    audio_file: str
    model: str
    reference: str
    hypothesis: str
    reference_normalized: str
    hypothesis_normalized: str
    wer: float
    duration_seconds: float


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


def get_available_audio_files():
    """Load audio files and transcriptions from metadata.csv."""
    audio_files = []

    with open(METADATA_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = DATASET_PATH / row['file_name']
            if audio_path.exists():
                audio_files.append({
                    "audio": audio_path,
                    "transcription": row['transcription'],
                    "name": audio_path.stem,
                    "category": row.get('category', '')
                })

    return audio_files


def transcribe_with_whisper_cli(audio_path: Path, model_path: Path) -> tuple[str, float]:
    """
    Transcribe audio using whisper-cli (whisper.cpp).
    Returns (transcription_text, duration_seconds).

    IMPORTANT: Runs synchronously - one at a time for accuracy.
    """
    start_time = time.time()

    # Create temp file for output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        output_base = f.name[:-4]  # Remove .txt extension

    cmd = [
        "whisper-cli",
        "-m", str(model_path),
        "-f", str(audio_path),
        "-l", "en",
        "-np",  # No prints except result
        "-nt",  # No timestamps
        "-of", output_base,
        "-otxt",  # Output as text
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per file
        )

        duration = time.time() - start_time

        # Read output file
        output_file = Path(f"{output_base}.txt")
        if output_file.exists():
            transcription = output_file.read_text().strip()
            output_file.unlink()  # Clean up
            return transcription, duration
        else:
            # Try reading from stdout as fallback
            return result.stdout.strip(), duration

    except subprocess.TimeoutExpired:
        return "[TIMEOUT]", time.time() - start_time
    except Exception as e:
        return f"[ERROR: {e}]", time.time() - start_time


def evaluate_model(model_name: str, model_path: Path, audio_files: list, normalizer) -> list[TranscriptionResult]:
    """
    Evaluate a single model on all audio files.
    Runs transcriptions ONE AT A TIME.
    """
    results = []
    total = len(audio_files)

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Model path: {model_path}")
    print(f"Files to process: {total}")
    print(f"{'='*60}")

    if not model_path.exists():
        print(f"  WARNING: Model not found at {model_path}")
        return results

    for i, file_info in enumerate(audio_files, 1):
        print(f"  [{i}/{total}] {file_info['name']}...", end=" ", flush=True)

        # Get reference text from metadata
        reference = file_info['transcription']

        # Transcribe (one at a time!)
        hypothesis, duration = transcribe_with_whisper_cli(file_info['audio'], model_path)

        # Normalize both for fair comparison
        ref_normalized = normalizer(reference)
        hyp_normalized = normalizer(hypothesis)

        # Calculate WER using werpy (requires list inputs)
        wer = werpy.wer([ref_normalized], [hyp_normalized])

        result = TranscriptionResult(
            audio_file=file_info['name'],
            model=model_name,
            reference=reference,
            hypothesis=hypothesis,
            reference_normalized=ref_normalized,
            hypothesis_normalized=hyp_normalized,
            wer=wer,
            duration_seconds=duration
        )
        results.append(result)

        print(f"WER: {wer:.2%} ({duration:.1f}s)")

    return results


def calculate_summary(model_name: str, results: list[TranscriptionResult]) -> ModelSummary:
    """Calculate summary statistics for a model's results."""
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

    wers = [r.wer for r in results]
    times = [r.duration_seconds for r in results]
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


def save_results(all_results: dict, summaries: list[ModelSummary]):
    """Save results to JSON and CSV files."""
    OUTPUT_PATH.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results as JSON
    json_path = OUTPUT_PATH / f"detailed_results_{timestamp}.json"
    json_data = {
        "timestamp": timestamp,
        "summaries": [asdict(s) for s in summaries],
        "results": {
            model: [asdict(r) for r in results]
            for model, results in all_results.items()
        }
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\nDetailed results saved to: {json_path}")

    # Save summary as CSV
    csv_path = OUTPUT_PATH / f"summary_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        for s in summaries:
            writer.writerow(asdict(s))
    print(f"Summary saved to: {csv_path}")

    # Save per-file WER comparison CSV
    comparison_path = OUTPUT_PATH / f"per_file_comparison_{timestamp}.csv"
    with open(comparison_path, 'w', newline='') as f:
        models = list(all_results.keys())
        fieldnames = ['audio_file'] + [f"{m}_wer" for m in models]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Get all audio files from first model
        if models:
            first_model_results = all_results[models[0]]
            for i, result in enumerate(first_model_results):
                row = {'audio_file': result.audio_file}
                for model in models:
                    if i < len(all_results[model]):
                        row[f"{model}_wer"] = f"{all_results[model][i].wer:.4f}"
                writer.writerow(row)
    print(f"Per-file comparison saved to: {comparison_path}")


def print_comparison_table(summaries: list[ModelSummary]):
    """Print a comparison table of all models."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    # Group by model size for comparison
    print(f"\n{'Model':<25} {'Avg WER':>10} {'Med WER':>10} {'Min':>8} {'Max':>8} {'Time/File':>10}")
    print("-"*80)

    # Sort by average WER
    for s in sorted(summaries, key=lambda x: x.average_wer):
        print(f"{s.model:<25} {s.average_wer:>9.2%} {s.median_wer:>9.2%} "
              f"{s.min_wer:>7.2%} {s.max_wer:>7.2%} {s.avg_time_per_file:>9.1f}s")

    # Show finetune vs original comparison
    print("\n" + "-"*80)
    print("FINETUNE VS ORIGINAL COMPARISON:")
    print("-"*80)

    sizes = ['tiny', 'small', 'medium', 'large']
    summary_dict = {s.model: s for s in summaries}

    for size in sizes:
        ft_key = f"finetune-{size}"
        orig_key = f"original-{size}"

        if ft_key in summary_dict and orig_key in summary_dict:
            ft_wer = summary_dict[ft_key].average_wer
            orig_wer = summary_dict[orig_key].average_wer
            improvement = orig_wer - ft_wer
            pct_improvement = (improvement / orig_wer * 100) if orig_wer > 0 else 0

            symbol = "✓" if improvement > 0 else "✗"
            print(f"  {size.upper():<8}: Finetune {ft_wer:.2%} vs Original {orig_wer:.2%} "
                  f"→ {symbol} {abs(improvement):.2%} {'better' if improvement > 0 else 'worse'} "
                  f"({abs(pct_improvement):.1f}% {'improvement' if improvement > 0 else 'regression'})")


def main():
    """Main evaluation entry point."""
    print("="*60)
    print("STT FINE-TUNE EVALUATION")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize normalizer
    normalizer = EnglishTextNormalizer()
    print("\nUsing EnglishTextNormalizer for fair comparison")
    print("  - Handles number formats (3000 ↔ three thousand)")
    print("  - Normalizes punctuation and casing")
    print("  - Converts British → American spelling")

    # Get audio files
    audio_files = get_available_audio_files()
    print(f"\nFound {len(audio_files)} audio files with matching ground truth")

    if not audio_files:
        print("ERROR: No audio files found!")
        return

    # Load and check available models
    print(f"\nLoading models from: {MODELS_CONFIG}")
    models = load_models_from_config()
    available_models = {}
    for name, path in models.items():
        if path.exists():
            print(f"  ✓ {name}: {path}")
            available_models[name] = path
        else:
            print(f"  ✗ {name}: NOT FOUND")

    if not available_models:
        print("ERROR: No models available!")
        return

    # Evaluate each model (save incrementally after each)
    OUTPUT_PATH.mkdir(exist_ok=True)
    all_results = {}
    summaries = []
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_name, model_path in available_models.items():
        results = evaluate_model(model_name, model_path, audio_files, normalizer)
        all_results[model_name] = results
        summary = calculate_summary(model_name, results)
        summaries.append(summary)

        # Save incrementally after each model
        incremental_path = OUTPUT_PATH / f"incremental_{run_timestamp}.json"
        with open(incremental_path, 'w') as f:
            json.dump({
                "timestamp": run_timestamp,
                "completed_models": list(all_results.keys()),
                "summaries": [asdict(s) for s in summaries],
                "results": {m: [asdict(r) for r in res] for m, res in all_results.items()}
            }, f, indent=2)
        print(f"  → Saved incremental results to {incremental_path}")

    # Print comparison and save final results
    print_comparison_table(summaries)
    save_results(all_results, summaries)

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

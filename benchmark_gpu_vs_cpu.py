#!/usr/bin/env python3
"""
CPU vs GPU Benchmark for whisper.cpp

Compares inference time between CPU and Vulkan GPU backends.
"""

import subprocess
import time
import json
import statistics
from pathlib import Path
from datetime import datetime

# Paths
SCRIPT_DIR = Path(__file__).parent
DATASET_PATH = Path.home() / "repos/hugging-face/datasets/private/Small-STT-Eval-Audio-Dataset/data"
OUTPUT_PATH = SCRIPT_DIR / "results"

# Model to benchmark (using medium as representative)
MODEL_PATH = Path("/home/daniel/ai/models/stt/openai-whisper/models/ggml-large-v3-turbo.bin")

# Binaries
WHISPER_CPU = Path.home() / ".local/bin/whisper-cli"
WHISPER_GPU = Path.home() / ".local/bin/whisper-cli-vulkan"

# Number of samples to benchmark
NUM_SAMPLES = 10


def get_system_info() -> dict:
    """Gather system information."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "os": "Ubuntu 25.10",
        "gpu": {
            "device": "AMD Radeon RX 7700 XT (RADV NAVI32)",
            "vulkan_api": "1.4.318",
            "driver_version": "25.2.3 (RADV)",
            "architecture": "RDNA 3 (gfx1101 / Navi 32)"
        },
        "cpu": {},
        "whisper_cpp": {
            "gpu_backend": "Vulkan (ggml-vulkan)",
            "build_flags": "-DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release"
        }
    }

    # Get CPU info
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    info["cpu"]["model"] = line.split(":")[1].strip()
                    break

        result = subprocess.run(["nproc"], capture_output=True, text=True)
        info["cpu"]["cores"] = int(result.stdout.strip())
    except:
        pass

    return info


def run_transcription(whisper_binary: Path, audio_path: Path, model_path: Path) -> float:
    """Run a single transcription and return duration in seconds."""
    cmd = [
        str(whisper_binary),
        "-m", str(model_path),
        "-f", str(audio_path),
        "-l", "en",
        "-np",
        "-nt",
    ]

    start = time.time()
    try:
        subprocess.run(cmd, capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        return -1
    return time.time() - start


def benchmark(backend_name: str, whisper_binary: Path, audio_files: list) -> dict:
    """Benchmark a whisper backend on multiple files."""
    print(f"\nBenchmarking {backend_name}...")
    print(f"Binary: {whisper_binary}")

    if not whisper_binary.exists():
        print(f"  ERROR: Binary not found!")
        return {"error": "binary not found"}

    times = []
    for i, audio_path in enumerate(audio_files):
        print(f"  [{i+1}/{len(audio_files)}] {audio_path.name}...", end=" ", flush=True)
        duration = run_transcription(whisper_binary, audio_path, MODEL_PATH)
        times.append(duration)
        print(f"{duration:.2f}s")

    return {
        "backend": backend_name,
        "model": MODEL_PATH.name,
        "num_samples": len(times),
        "times_seconds": times,
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times),
        "total": sum(times)
    }


def main():
    print("=" * 60)
    print("CPU vs GPU BENCHMARK")
    print("=" * 60)

    # Get audio files
    audio_files = sorted(DATASET_PATH.glob("*.wav"))[:NUM_SAMPLES]
    print(f"\nUsing {len(audio_files)} audio samples")
    print(f"Model: {MODEL_PATH.name}")

    # Gather system info
    system_info = get_system_info()
    print(f"\nSystem: {system_info['os']}")
    print(f"CPU: {system_info['cpu'].get('model', 'Unknown')}")
    print(f"GPU: {system_info['gpu']['device']}")

    results = {
        "system": system_info,
        "benchmarks": {}
    }

    # Benchmark GPU first (faster)
    gpu_results = benchmark("Vulkan GPU", WHISPER_GPU, audio_files)
    results["benchmarks"]["gpu"] = gpu_results

    # Benchmark CPU
    cpu_results = benchmark("CPU", WHISPER_CPU, audio_files)
    results["benchmarks"]["cpu"] = cpu_results

    # Calculate speedup
    if cpu_results.get("mean") and gpu_results.get("mean"):
        speedup = cpu_results["mean"] / gpu_results["mean"]
        results["speedup"] = {
            "factor": speedup,
            "description": f"GPU is {speedup:.1f}x faster than CPU"
        }

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nModel: {MODEL_PATH.name}")
    print(f"Samples: {NUM_SAMPLES}")
    print(f"\n{'Backend':<15} {'Mean':>10} {'Median':>10} {'Stdev':>10}")
    print("-" * 50)

    for name, data in [("CPU", cpu_results), ("GPU (Vulkan)", gpu_results)]:
        if "error" not in data:
            print(f"{name:<15} {data['mean']:>9.2f}s {data['median']:>9.2f}s {data['stdev']:>9.2f}s")

    if "speedup" in results:
        print(f"\n>>> {results['speedup']['description']} <<<")

    # Save results
    OUTPUT_PATH.mkdir(exist_ok=True)
    output_file = OUTPUT_PATH / f"benchmark_cpu_vs_gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

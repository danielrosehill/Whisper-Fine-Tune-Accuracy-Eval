# Evaluation Scripts

Tools used to run the STT fine-tune evaluation.

## Scripts

| Script | Purpose |
|--------|---------|
| `run_gui.sh` | Main launcher - runs the GUI evaluation tool |
| `gui_evaluate.py` | PyQt6 GUI for running evaluations with progress tracking |
| `evaluate.py` | Core evaluation logic (WER calculation, transcription) |
| `recorder.py` | Audio recording utility for creating test samples |
| `benchmark_gpu_vs_cpu.py` | Benchmarking script for GPU vs CPU performance |
| `validate_results.py` | Validates and cleans result data |
| `run.sh` | Simple launcher for recorder |

## Usage

From the repository root:

```bash
./scripts/run_gui.sh
```

## Requirements

- Python 3.10+
- Virtual environment with dependencies from `requirements.txt`
- `whisper-cli-vulkan` (whisper.cpp with Vulkan support) in PATH

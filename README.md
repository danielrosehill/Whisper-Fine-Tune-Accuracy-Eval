# Fine-Tune-Accuracy-Evaluation

Evaluates fine-tuned Whisper STT models against original OpenAI models using Word Error Rate (WER).

## Quick Start

```bash
./run_gui.sh
```

## Methodology

### WER Calculation with werpy

Uses [werpy](https://github.com/analyticsinmotion/werpy) for Word Error Rate calculation. WER measures the edit distance between the reference transcription and the hypothesis (model output), normalized by the number of words in the reference.

```python
import werpy
wer = werpy.wer([reference_normalized], [hypothesis_normalized])
```

### Text Normalization with whisper-normalizer

Uses [whisper-normalizer](https://github.com/kurianbenoy/whisper_normalizer) for fair comparison between reference and hypothesis text. This is critical because STT models may output semantically correct but textually different transcriptions.

```python
from whisper_normalizer.english import EnglishTextNormalizer
normalizer = EnglishTextNormalizer()

# Both reference and hypothesis are normalized before WER calculation
ref_normalized = normalizer(reference)
hyp_normalized = normalizer(hypothesis)
```

**What the normalizer handles:**

| Input | Normalized Output |
|-------|-------------------|
| "3000" | "three thousand" |
| "three thousand" | "three thousand" |
| "Mr. Smith" | "mister smith" |
| "It's 5:30 PM" | "its five thirty pm" |
| "colour" (British) | "color" (American) |
| "HELLO WORLD" | "hello world" |

This approach ensures that a model outputting "3000" and one outputting "three thousand" are scored equally when the reference contains either form.

### Why Runtime Normalization?

The normalization is applied at evaluation time, not pre-applied to the dataset. This:
- Preserves the original reference transcriptions
- Allows fair comparison regardless of how numbers/abbreviations appear in ground truth
- Makes the evaluation methodology reproducible

## Models Compared

**Fine-tuned (GGML format)**:
- tiny, base, small, medium, large-v3-turbo

**Original OpenAI (GGML format)**:
- tiny, base, small, medium, large-v3-turbo

Model paths configured in `models.json`.

## Transcription Engine

Uses `whisper-cli-vulkan` ([whisper.cpp](https://github.com/ggerganov/whisper.cpp)) with **Vulkan GPU acceleration** for transcription.

### GPU Acceleration

The evaluation uses Vulkan backend for GPU-accelerated inference, providing ~20x speedup over CPU:

| Backend | large-v3-turbo | small |
|---------|----------------|-------|
| CPU     | ~22s/file      | ~5s/file |
| Vulkan GPU | ~1s/file    | <0.5s/file |

### Building whisper.cpp with Vulkan

```bash
cd ~/programs/ai-ml/whisper.cpp
mkdir build-vulkan && cd build-vulkan
cmake .. -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
make -j12
cp bin/whisper-cli ~/.local/bin/whisper-cli-vulkan
```

**Requirements:**
- Vulkan SDK and drivers
- AMD GPU with RADV driver (tested on RX 7700 XT)
- Or NVIDIA GPU with appropriate Vulkan support

### Verifying GPU Usage

When running, you should see:
```
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon RX 7700 XT (RADV NAVI32)
whisper_backend_init_gpu: using Vulkan0 backend
```

If you see `whisper_backend_init_gpu: no GPU found`, the binary was compiled without GPU support.

## GUI Features

- **Progress monitoring**: Real-time progress bars and log output
- **Incremental saving**: Each transcription saved as individual JSON file
- **Resume capability**: Skips already-completed files on restart
- **Results table**: Color-coded WER summary
- **Comparison view**: Side-by-side fine-tune vs original analysis

## Output Structure

```
results/
├── individual/                          # Per-transcription results
│   ├── finetune-tiny/
│   │   ├── 001_tech_github.json
│   │   ├── 002_tech_github.json
│   │   └── ...
│   ├── finetune-base/
│   │   └── ...
│   └── original-tiny/
│       └── ...
├── evaluation_progress.json             # Aggregate progress
├── detailed_results_YYYYMMDD_HHMMSS.json
└── summary_YYYYMMDD_HHMMSS.csv
```

Each individual result JSON contains:
```json
{
  "audio_file": "001_tech_github",
  "model": "finetune-base",
  "reference": "I pushed the changes to GitHub and opened a pull request against the main branch. The CI pipeline is running now and should finish in a few minutes.",
  "hypothesis": "I poached the changes to GitHub and opened a pulled request against the main branch. The CI pipeline is running now and should finish in a few minutes.",
  "reference_normalized": "i pushed the changes to github and opened a pull request against the main branch the ci pipeline is running now and should finish in a few minutes",
  "hypothesis_normalized": "i poached the changes to github and opened a pulled request against the main branch the ci pipeline is running now and should finish in a few minutes",
  "wer": 0.07142857142857142,
  "duration_seconds": 1.3955373764038086
}
```

In this example, the model made 2 errors ("pushed"→"poached", "pull"→"pulled") out of 28 words = 7.14% WER.

## Dependencies

```
werpy
whisper-normalizer
PyQt6
numpy
```

## Performance Notes

- **Sequential execution**: Transcriptions run one at a time for accurate timing measurements
- **Inference time recorded**: Each result includes `duration_seconds` for benchmarking
- **GPU memory**: Large-v3-turbo uses ~1.6GB VRAM
- **Expected runtime**: Full evaluation (10 models × 92 files) completes in ~10-15 minutes with GPU

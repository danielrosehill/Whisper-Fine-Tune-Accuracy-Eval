# Fine-Tune Accuracy Evaluation

Evaluates fine-tuned Whisper STT models against original OpenAI models using Word Error Rate (WER).

## Results

**Test:** 91 audio samples with technical vocabulary (GitHub, Hugging Face, Docker, cloud technologies)
**Full analysis:** [RESULTS.md](RESULTS.md)

### Word Error Rate Comparison

| Model Size | Fine-tuned | Original | Improvement |
|------------|------------|----------|-------------|
| **tiny** | 6.59% | 7.85% | **+16% better** |
| **base** | 5.02% | 5.46% | **+8% better** |
| **small** | 3.45% | 3.96% | **+13% better** |
| **medium** | 2.96% | 2.71% | -9% worse |
| **large-v3-turbo** | 3.01% | 2.77% | -9% worse |

### Key Findings

- **Fine-tuning improves smaller models significantly** - tiny, base, and small all show meaningful WER reductions
- **Larger models don't benefit from fine-tuning** - medium and large perform slightly better without it
- **Best value:** Fine-tuned small model (3.45% WER, ~0.8s inference)
- **Best accuracy:** Original medium model (2.71% WER)

![WER Comparison](visualizations/wer_comparison.png)
![Inference Time Comparison](visualizations/inference_time_comparison.png)

## Models Compared

**Fine-tuned (GGML format):** tiny, base, small, medium, large-v3-turbo
**Original OpenAI (GGML format):** tiny, base, small, medium, large-v3-turbo

Fine-tuned models: [danielrosehill/whisper-finetunes](https://huggingface.co/danielrosehill)

## Methodology

### WER Calculation

Uses [werpy](https://github.com/analyticsinmotion/werpy) for Word Error Rate calculation with [whisper-normalizer](https://github.com/kurianbenoy/whisper_normalizer) for text normalization:

| Input | Normalized Output |
|-------|-------------------|
| "3000" | "three thousand" |
| "Mr. Smith" | "mister smith" |
| "It's 5:30 PM" | "its five thirty pm" |

This ensures fair comparison regardless of how numbers/abbreviations appear in the transcription.

### Transcription Engine

Uses `whisper-cli-vulkan` ([whisper.cpp](https://github.com/ggerganov/whisper.cpp)) with Vulkan GPU acceleration (~20x faster than CPU).

## Running the Evaluation

```bash
./scripts/run_gui.sh
```

See [scripts/README.md](scripts/README.md) for details on the evaluation tools.

## Output Structure

```
results/
├── individual/                          # Per-transcription results
│   ├── finetune-tiny/
│   ├── finetune-base/
│   └── original-tiny/
├── detailed_results_YYYYMMDD_HHMMSS.json
└── summary_YYYYMMDD_HHMMSS.csv
```

## Dependencies

```
werpy
whisper-normalizer
PyQt6
numpy
```

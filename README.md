# Does Fine-Tuning Whisper Reduce Word Error Rate / Improve Accuracy?

Fine tuning ASR models is commonly undetaken for several purposes:

- Improve overall accuracy (or conversely, reduce WER) 
- Improve accuracy when transcribing specialist vocabulary by training on a subset of domain specialist dense vocab texts 
- Improving accuracy for languages with small speaker populations and which are underrepresented in ASR training data (with impaired accuracy as as result) 

This evaluation used local inference to compare fine-tunes I trained on about 90 minutes of my speech. The targeted improvements were in tech vocab recognition and enhanced performance in "code switching" (English with Hebrew words).

The results were surprising but potentially highly relevant:

- The smaller Whisper models showed a definite improvement in accuracy with fine-tuning with a very modest "penalty" in the form of increased inference time 
- The larger models (medium and large) surprisingly showed decreased accuracy / increased WER 
- This valuation was conducted on GPU accelerated inference.  In this environment, the differences in inference time between fine tunes and originals were not significant. 

## About This Evaluation

**Important context:** The fine-tuned models tested here were **proof-of-concept efforts** trained on approximately **90 minutes of audio data**. 

These results may not be representative of what a more substantial fine-tuning effort (with significantly more training data and hyperparameter optimization) could achieve.

That said, the findings are instructive—particularly for understanding where fine-tuning may offer the most value, such as improving WER on **edge devices and mobile applications** where smaller, faster models are preferred.

### Method

- Evaluation dataset with ~100 sentences designed to test ability of models to transcribe specialist text and code-switching samples accurately
- Text normalization with Whisper Normalizer for fair comparison
- GPU-accelerated evaluation using Vulkan

## Results

**Test:** 91 audio samples with technical vocabulary (GitHub, Hugging Face, Docker, cloud technologies)
**Full analysis:** [RESULTS.md](RESULTS.md)

### Word Error Rate Comparison

| Model Size | Fine-tuned | Original | Result |
|------------|------------|----------|--------|
| **tiny** | 6.59% | 7.85% | **Fine-tune better (+16%)** |
| **base** | 5.02% | 5.46% | **Fine-tune better (+8%)** |
| **small** | 3.45% | 3.96% | **Fine-tune better (+13%)** |
| **medium** | 2.96% | 2.71% | Fine-tune worse (-9%) |
| **large-v3-turbo** | 3.01% | 2.77% | Fine-tune worse (-9%) |

### Inference Time Comparison

| Model Size | Fine-tuned | Original | Difference |
|------------|------------|----------|------------|
| **tiny** | 0.247s | 0.275s | -0.028s (10% faster) |
| **base** | 0.408s | 0.413s | -0.005s (similar) |
| **small** | 0.823s | 0.518s | +0.305s (59% slower) |
| **medium** | 1.462s | 1.084s | +0.378s (35% slower) |
| **large-v3-turbo** | 0.991s | 1.004s | -0.013s (similar) |

*Note: Inference time variations for fine-tuned models may be due to quantization differences in GGML conversion.*

### Key Findings

- **Fine-tuning improves smaller models significantly** - tiny, base, and small all show meaningful WER reductions (8-16%)
- **Larger models don't benefit from this fine-tune** - medium and large-v3-turbo performed slightly worse with fine-tuning
- **Promising for edge/mobile deployment** - The smaller models (tiny, base, small) that benefit most from fine-tuning are also the models most suitable for resource-constrained environments
- **Best value:** Fine-tuned small model (3.45% WER, ~0.8s inference)
- **Best accuracy:** Original medium model (2.71% WER)

![Fine-Tuning Impact](visualizations/finetune_improvement.png)

![WER Comparison](visualizations/wer_comparison.png)
![Inference Time Comparison](visualizations/inference_time_comparison.png)

## Models Compared

**Fine-tuned (GGML format):** tiny, base, small, medium, large-v3-turbo
**Original OpenAI (GGML format):** tiny, base, small, medium, large-v3-turbo

Fine-tuned models: [danielrosehill/whisper-finetunes](https://huggingface.co/danielrosehill)

## Datasets

- **Audio dataset:** [Small-STT-Eval-Audio-Dataset](https://huggingface.co/datasets/danielrosehill/Small-STT-Eval-Audio-Dataset)
- **Evaluation results:** [STT-Fine-Tune-Eval-101225](https://huggingface.co/datasets/danielrosehill/STT-Fine-Tune-Eval-101225) (also in repo)

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

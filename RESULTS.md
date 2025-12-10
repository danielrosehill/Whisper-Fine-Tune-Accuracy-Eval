# Evaluation Results

**Date:** December 10, 2025
**Test Samples:** 91 audio files (1 excluded due to complete transcription failure across all models)
**Inference:** Vulkan GPU acceleration (AMD Radeon RX 7700 XT)

## Summary

Fine-tuned Whisper models were compared against their corresponding OpenAI base models across 5 model sizes. The fine-tuning was performed on domain-specific technical vocabulary (GitHub, Hugging Face, Docker, cloud technologies, etc.).

## Head-to-Head Comparison

### Word Error Rate (WER)

| Model Size | Fine-tuned | Original | Difference | Winner |
|------------|------------|----------|------------|--------|
| **tiny** | 6.59% | 7.85% | -1.26% | Fine-tuned |
| **base** | 5.02% | 5.46% | -0.44% | Fine-tuned |
| **small** | 3.45% | 3.96% | -0.51% | Fine-tuned |
| **medium** | 2.96% | 2.71% | +0.25% | Original |
| **large-v3-turbo** | 3.01% | 2.77% | +0.24% | Original |

**Key Finding:** Fine-tuning provides meaningful accuracy improvements for smaller models (tiny, base, small), but larger models (medium, large) perform slightly better without fine-tuning on this test set.

### Inference Time (seconds per file)

| Model Size | Fine-tuned | Original | Difference |
|------------|------------|----------|------------|
| **tiny** | 0.247s | 0.275s | -0.028s (10% faster) |
| **base** | 0.408s | 0.413s | -0.005s (similar) |
| **small** | 0.823s | 0.518s | +0.305s (59% slower) |
| **medium** | 1.462s | 1.084s | +0.378s (35% slower) |
| **large-v3-turbo** | 0.991s | 1.004s | -0.013s (similar) |

**Note:** Inference time variations for fine-tuned models may be due to quantization differences in GGML conversion.

## Detailed Results by Model

### Fine-tuned Models

| Model | Avg WER | Median WER | Min WER | Max WER | Avg Time |
|-------|---------|------------|---------|---------|----------|
| tiny | 6.59% | 4.55% | 0.00% | 33.33% | 0.247s |
| base | 5.02% | 3.85% | 0.00% | 20.83% | 0.408s |
| small | 3.45% | 0.00% | 0.00% | 19.23% | 0.823s |
| medium | 2.96% | 0.00% | 0.00% | 19.05% | 1.462s |
| large-v3-turbo | 3.01% | 0.00% | 0.00% | 18.18% | 0.991s |

### Original OpenAI Models

| Model | Avg WER | Median WER | Min WER | Max WER | Avg Time |
|-------|---------|------------|---------|---------|----------|
| tiny | 7.85% | 4.76% | 0.00% | 26.32% | 0.275s |
| base | 5.46% | 4.17% | 0.00% | 21.05% | 0.413s |
| small | 3.96% | 0.00% | 0.00% | 19.05% | 0.518s |
| medium | 2.71% | 0.00% | 0.00% | 14.29% | 1.084s |
| large-v3-turbo | 2.77% | 0.00% | 0.00% | 15.79% | 1.004s |

## Conclusions

1. **Fine-tuning is most effective for smaller models.** The tiny model saw a 16% relative improvement in WER (7.85% to 6.59%), making fine-tuning worthwhile when model size/speed is a constraint.

2. **Larger models may not benefit from fine-tuning.** Medium and large models performed marginally better without fine-tuning, suggesting they already have sufficient capacity to handle the technical vocabulary.

3. **Best accuracy/speed tradeoff:** The fine-tuned small model offers an excellent balance at 3.45% WER with ~0.8s inference time.

4. **For maximum accuracy:** The original medium model achieved the lowest WER (2.71%) while being faster than the fine-tuned medium.

## Methodology

- **WER Calculation:** Using [werpy](https://github.com/analyticsinmotion/werpy)
- **Text Normalization:** Using [whisper-normalizer](https://github.com/kurianbenoy/whisper_normalizer) to handle number formats, abbreviations, and spelling variations
- **Transcription Engine:** whisper.cpp with Vulkan GPU acceleration
- **Model Format:** GGML (whisper.cpp compatible)

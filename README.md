# Fine-Tune-Accuracy-Evaluation

Evaluates fine-tuned Whisper STT models against original OpenAI models.

## Methodology

- **WER Calculation**: Uses [werpy](https://github.com/analyticsinmotion/werpy) for Word Error Rate
- **Text Normalization**: Uses [whisper-normalizer](https://github.com/kurianbenoy/whisper_normalizer) at runtime (not pre-applied to dataset)
  - Handles number format ambiguity ("3000" ↔ "three thousand")
  - Normalizes punctuation and casing
  - Converts British → American spelling
- **Sequential Execution**: Transcriptions run one at a time for accurate timing/resource measurements

## Models Compared

**Fine-tuned (GGML)**:
- tiny, base, small, medium, large

**Original OpenAI (GGML)**:
- tiny, small, medium, large-v3-turbo

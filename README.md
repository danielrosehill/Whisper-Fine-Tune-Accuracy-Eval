# Does Fine-Tuning Whisper Improve Accuracy?

**TL;DR:** Fine-tuning improved smaller models but **made larger models worse**. The exception: code-switching (mixing languages mid-sentence) improved dramatically across all model sizes.

## The Surprising Finding: Larger Models Got Worse

Fine-tuning on ~90 minutes of audio produced unexpected results:

| Model Size | Fine-tuned WER | Original WER | Result |
|------------|----------------|--------------|--------|
| tiny | 6.59% | 7.85% | **+16% better** |
| base | 5.02% | 5.46% | **+8% better** |
| small | 3.45% | 3.96% | **+13% better** |
| medium | 2.96% | 2.71% | **-9% worse** |
| large | 3.01% | 2.77% | **-9% worse** |

![Fine-Tuning Impact](visualizations/finetune_improvement.png)

### Why Did Larger Models Degrade?

Several plausible explanations:

1. **Capacity vs. data mismatch** - Larger models have more parameters to update. With only 90 minutes of training data, the fine-tuning may have introduced noise rather than signal, causing the model to "forget" some of its original capabilities.

2. **Already optimized** - Medium and large models likely already handle technical vocabulary well from their massive pre-training. Fine-tuning on limited domain-specific data may have narrowed their generalization.

3. **Overfitting to speaker characteristics** - The fine-tuning data came from a single speaker. Larger models may have overfit to idiosyncratic speech patterns that don't generalize to the test set.

4. **Catastrophic forgetting** - A known phenomenon where fine-tuning causes neural networks to lose previously learned information. Larger models with more parameters may be more susceptible when training data is limited.

**Implication:** Don't assume fine-tuning will help. For larger models, you may need significantly more training data (hours, not minutes) to see benefits.

---

## The Code-Switching Win

### What is Code-Switching?

Code-switching is when speakers mix two or more languages within a conversation or even mid-sentence. Examples:

- **Spanish/English (Spanglish):** "I'm going to the store para comprar some milk"
- **Hindi/English (Hinglish):** "That meeting was bahut boring yaar"
- **Hebrew/English:** "Can you send me the document b'vakasha?"

This is extremely common among:
- Bilingual immigrant communities (Hispanic Americans, South Asian diaspora)
- Multilingual regions (India, Philippines, North Africa)
- Second-generation speakers who grew up with two languages

Standard ASR models often struggle with code-switching because they're typically trained on monolingual data.

### Fine-Tuning Dramatically Improved Code-Switching

Unlike the mixed results for technical vocabulary, fine-tuning improved code-switched content across **all model sizes**:

| Model | Fine-tuned | Original | Improvement |
|-------|------------|----------|-------------|
| tiny | 9.99% | 13.50% | **+26%** |
| base | 7.40% | 10.01% | **+26%** |
| small | 2.71% | 6.69% | **+60%** |
| medium | 4.47% | 6.24% | **+28%** |
| large | 1.92% | 5.00% | **+62%** |

![Code-Switching Comparison](analysis/visualizations/hebrew_codeswitching_comparison.png)

### Why Does Fine-Tuning Help Code-Switching So Much?

1. **Underrepresented in training data** - Pre-trained models see mostly monolingual data. Even a small amount of code-switched examples teaches the model this pattern exists.

2. **Acoustic adaptation** - The model learns how a speaker transitions between languages, including accent shifts and pronunciation patterns.

3. **Vocabulary bridging** - Fine-tuning teaches the model that foreign words can appear in English contexts, reducing the tendency to force English interpretations.

**Implication:** If you're building ASR for bilingual communities (Spanish/English in the US, Hindi/English in India, etc.), fine-tuning on even a modest amount of code-switched data could yield significant improvements.

---

## Practical Recommendations

### Decision Matrix

| Use Case | Recommendation |
|----------|----------------|
| **Edge/mobile deployment** | Fine-tune small model (best accuracy/speed trade-off) |
| **Maximum accuracy on English** | Use original medium (2.71% WER) |
| **Bilingual/code-switched content** | Fine-tune any model size |
| **Fastest inference** | Fine-tune tiny (0.25s, 16% better than original) |
| **Limited fine-tuning data (<2 hours)** | Stick to smaller models (tiny, base, small) |

### When to Fine-Tune

**Do fine-tune when:**
- Targeting smaller models for resource-constrained environments
- Users frequently code-switch between languages
- Transcribing underrepresented languages or dialects
- Speaker has distinctive pronunciation patterns

**Skip fine-tuning when:**
- Using larger models on well-represented content
- You have limited training data and need medium/large accuracy
- Technical vocabulary is already well-recognized by base models

---

## Detailed Results

### Word Error Rate Comparison
![WER Comparison](visualizations/wer_comparison.png)

### Inference Time (GPU)
![Inference Time](visualizations/inference_time_comparison.png)

| Model Size | Fine-tuned | Original |
|------------|------------|----------|
| tiny | 0.247s | 0.275s |
| base | 0.408s | 0.413s |
| small | 0.823s | 0.518s |
| medium | 1.462s | 1.084s |
| large | 0.991s | 1.004s |

*Note: GPU inference minimizes speed differences. CPU inference may show larger variations.*

### Technical Vocabulary Results

Fine-tuning helped smaller models but hurt larger ones:

| Model | Fine-tuned | Original | Result |
|-------|------------|----------|--------|
| tiny | 6.71% | 7.49% | +10% better |
| base | 5.29% | 6.65% | +20% better |
| small | 3.37% | 3.49% | +4% better |
| medium | 2.95% | 2.02% | -46% worse |
| large | 3.28% | 2.57% | -27% worse |

![Technical Vocabulary](analysis/visualizations/tech_vocab_comparison.png)
![Improvement Heatmap](analysis/visualizations/improvement_heatmap.png)

---

## Methodology

- **Test set:** 91 audio samples with technical vocabulary and code-switching
- **Normalization:** Whisper Normalizer for fair comparison
- **WER calculation:** [werpy](https://github.com/analyticsinmotion/werpy)
- **Inference:** whisper.cpp with Vulkan GPU acceleration
- **Hardware:** AMD Radeon RX 7700 XT

### Limitations

This was a proof-of-concept with ~90 minutes of single-speaker training data. Results may improve with:
- More training data (hours instead of minutes)
- Hyperparameter optimization
- Multi-speaker datasets
- Data augmentation

---

## Resources

- **Fine-tuned models:** [danielrosehill/whisper-finetunes](https://huggingface.co/danielrosehill)
- **Audio dataset:** [Small-STT-Eval-Audio-Dataset](https://huggingface.co/datasets/danielrosehill/Small-STT-Eval-Audio-Dataset)
- **Results dataset:** [STT-Fine-Tune-Eval-101225](https://huggingface.co/datasets/danielrosehill/STT-Fine-Tune-Eval-101225)

## Running the Evaluation

```bash
./scripts/run_gui.sh
```

See [scripts/README.md](scripts/README.md) for details.

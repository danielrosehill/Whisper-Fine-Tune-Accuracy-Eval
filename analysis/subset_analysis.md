# Fine-Tuning Impact on Specialized Content

## Technical Vocabulary Analysis

| Model Size | Fine-tuned WER | Original WER | Improvement | Result |
|------------|----------------|--------------|-------------|--------|
| tiny | 6.71% | 7.49% | +10.4% | **Fine-tune better** |
| base | 5.29% | 6.65% | +20.4% | **Fine-tune better** |
| small | 3.37% | 3.49% | +3.6% | **Fine-tune better** |
| medium | 2.95% | 2.02% | -45.9% | Fine-tune worse |
| large | 3.28% | 2.57% | -27.4% | Fine-tune worse |

## Hebrew Code-Switching Analysis

| Model Size | Fine-tuned WER | Original WER | Improvement | Result |
|------------|----------------|--------------|-------------|--------|
| tiny | 9.99% | 13.50% | +26.0% | **Fine-tune better** |
| base | 7.40% | 10.01% | +26.1% | **Fine-tune better** |
| small | 2.71% | 6.69% | +59.5% | **Fine-tune better** |
| medium | 4.47% | 6.24% | +28.4% | **Fine-tune better** |
| large | 1.92% | 5.00% | +61.6% | **Fine-tune better** |
## Detailed Category Breakdown

### Technical Vocabulary by Category


#### Tech Github

| Model Size | Fine-tuned | Original | Diff |
|------------|------------|----------|------|
| tiny | 8.05% | 12.69% | -4.64% |
| base | 3.13% | 5.04% | -1.90% |
| small | 1.58% | 0.87% | +0.71% |
| medium | 1.70% | 2.54% | -0.83% |
| large | 0.87% | 0.87% | +0.00% |

#### Tech Huggingface

| Model Size | Fine-tuned | Original | Diff |
|------------|------------|----------|------|
| tiny | 5.26% | 7.51% | -2.25% |
| base | 5.33% | 2.37% | +2.96% |
| small | 3.11% | 6.13% | -3.02% |
| medium | 5.34% | 2.37% | +2.97% |
| large | 3.06% | 3.91% | -0.85% |

#### Tech Docker

| Model Size | Fine-tuned | Original | Diff |
|------------|------------|----------|------|
| tiny | 2.56% | 3.64% | -1.08% |
| base | 0.91% | 21.82% | -20.91% |
| small | 2.45% | 0.91% | +1.54% |
| medium | 0.00% | 0.91% | -0.91% |
| large | 0.91% | 0.91% | +0.00% |

#### Tech Linux

| Model Size | Fine-tuned | Original | Diff |
|------------|------------|----------|------|
| tiny | 6.99% | 8.44% | -1.45% |
| base | 4.92% | 5.41% | -0.49% |
| small | 3.23% | 6.37% | -3.13% |
| medium | 2.28% | 2.44% | -0.17% |
| large | 1.72% | 3.00% | -1.28% |

#### Tech Api

| Model Size | Fine-tuned | Original | Diff |
|------------|------------|----------|------|
| tiny | 2.17% | 0.00% | +2.17% |
| base | 3.31% | 1.14% | +2.17% |
| small | 0.00% | 1.14% | -1.14% |
| medium | 0.00% | 0.00% | +0.00% |
| large | 2.17% | 2.17% | +0.00% |

#### Tech Python

| Model Size | Fine-tuned | Original | Diff |
|------------|------------|----------|------|
| tiny | 12.49% | 10.67% | +1.82% |
| base | 8.93% | 8.93% | +0.00% |
| small | 5.38% | 3.38% | +2.01% |
| medium | 6.42% | 2.61% | +3.81% |
| large | 7.20% | 2.77% | +4.43% |

#### Tech Web

| Model Size | Fine-tuned | Original | Diff |
|------------|------------|----------|------|
| tiny | 7.27% | 7.50% | -0.23% |
| base | 9.77% | 0.00% | +9.77% |
| small | 6.82% | 2.50% | +4.32% |
| medium | 4.55% | 0.00% | +4.55% |
| large | 0.00% | 0.00% | +0.00% |

#### Ai Ml

| Model Size | Fine-tuned | Original | Diff |
|------------|------------|----------|------|
| tiny | 3.01% | 4.10% | -1.09% |
| base | 3.15% | 2.24% | +0.91% |
| small | 1.23% | 1.27% | -0.04% |
| medium | 0.85% | 0.85% | +0.00% |
| large | 1.81% | 1.38% | +0.44% |

#### Local Tools

| Model Size | Fine-tuned | Original | Diff |
|------------|------------|----------|------|
| tiny | 11.81% | 11.47% | +0.34% |
| base | 9.65% | 9.75% | -0.09% |
| small | 7.29% | 6.93% | +0.36% |
| medium | 5.84% | 4.32% | +1.52% |
| large | 8.13% | 5.58% | +2.55% |

### Hebrew Code-Switching by Category


#### Hebrew Daily

| Model Size | Fine-tuned | Original | Diff |
|------------|------------|----------|------|
| tiny | 6.62% | 13.09% | -6.47% |
| base | 7.26% | 10.73% | -3.46% |
| small | 1.01% | 7.48% | -6.47% |
| medium | 3.37% | 6.64% | -3.27% |
| large | 1.57% | 5.77% | -4.20% |

#### Hebrew Food

| Model Size | Fine-tuned | Original | Diff |
|------------|------------|----------|------|
| tiny | 10.70% | 13.42% | -2.72% |
| base | 6.07% | 9.42% | -3.35% |
| small | 4.11% | 5.50% | -1.39% |
| medium | 1.33% | 6.02% | -4.68% |
| large | 1.33% | 2.72% | -1.39% |

#### Mixed Locale

| Model Size | Fine-tuned | Original | Diff |
|------------|------------|----------|------|
| tiny | 25.76% | 15.66% | +10.10% |
| base | 10.10% | 7.32% | +2.78% |
| small | 9.09% | 4.55% | +4.55% |
| medium | 14.65% | 4.55% | +10.10% |
| large | 4.55% | 4.55% | +0.00% |

## Key Findings

- **Technical Vocabulary**: Fine-tuning improved 3/5 model sizes
- **Hebrew Code-Switching**: Fine-tuning improved 5/5 model sizes

### Best Performing Fine-tuned Models:
- **Technical Vocabulary**: medium (2.95% WER)
- **Hebrew Code-Switching**: large (1.92% WER)

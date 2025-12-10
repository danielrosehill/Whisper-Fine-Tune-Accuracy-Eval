#!/usr/bin/env python3
"""
Analyze fine-tuning impact on technical vocabulary and Hebrew code-switching samples.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Define category groups
TECH_VOCAB_CATEGORIES = [
    'tech_github', 'tech_huggingface', 'tech_docker', 'tech_linux',
    'tech_api', 'tech_python', 'tech_web', 'ai_ml', 'local_tools'
]

HEBREW_CODESWITCHING_CATEGORIES = [
    'hebrew_daily', 'hebrew_food', 'mixed_locale'
]

RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'individual'
ANALYSIS_DIR = Path(__file__).parent.parent / 'analysis'
VIZ_DIR = ANALYSIS_DIR / 'visualizations'

MODEL_SIZES = ['tiny', 'base', 'small', 'medium', 'large']


def get_category_from_filename(filename: str) -> str:
    """Extract category from filename like '001_tech_github.json'"""
    parts = filename.replace('.json', '').split('_', 1)
    if len(parts) > 1:
        return parts[1]
    return ''


def load_results() -> dict:
    """Load all individual results into a structured dict."""
    results = defaultdict(lambda: defaultdict(list))

    for model_dir in RESULTS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for json_file in model_dir.glob('*.json'):
            category = get_category_from_filename(json_file.name)
            with open(json_file) as f:
                data = json.load(f)
                data['category'] = category
                results[model_name][category].append(data)

    return results


def calculate_category_wer(results: dict, categories: list) -> dict:
    """Calculate average WER for specific categories."""
    category_wer = defaultdict(lambda: defaultdict(list))

    for model_name, model_results in results.items():
        for category, samples in model_results.items():
            if category in categories:
                for sample in samples:
                    category_wer[model_name][category].append(sample['wer'])

    # Calculate averages
    averages = {}
    for model_name in category_wer:
        averages[model_name] = {}
        for category, wers in category_wer[model_name].items():
            averages[model_name][category] = {
                'mean': np.mean(wers) if wers else 0,
                'std': np.std(wers) if wers else 0,
                'count': len(wers),
                'values': wers
            }
        # Calculate overall for this subset
        all_wers = []
        for cat in categories:
            if cat in category_wer[model_name]:
                all_wers.extend(category_wer[model_name][cat])
        averages[model_name]['_overall'] = {
            'mean': np.mean(all_wers) if all_wers else 0,
            'std': np.std(all_wers) if all_wers else 0,
            'count': len(all_wers),
            'values': all_wers
        }

    return averages


def create_comparison_table(tech_wer: dict, hebrew_wer: dict) -> str:
    """Create markdown comparison table."""
    lines = []

    lines.append("# Fine-Tuning Impact on Specialized Content\n")
    lines.append("## Technical Vocabulary Analysis\n")
    lines.append("| Model Size | Fine-tuned WER | Original WER | Improvement | Result |")
    lines.append("|------------|----------------|--------------|-------------|--------|")

    for size in MODEL_SIZES:
        ft_key = f'finetune-{size}'
        orig_key = f'original-{size}'

        ft_wer = tech_wer.get(ft_key, {}).get('_overall', {}).get('mean', 0) * 100
        orig_wer = tech_wer.get(orig_key, {}).get('_overall', {}).get('mean', 0) * 100

        if orig_wer > 0:
            improvement = ((orig_wer - ft_wer) / orig_wer) * 100
            result = "**Fine-tune better**" if improvement > 0 else "Fine-tune worse"
        else:
            improvement = 0
            result = "N/A"

        lines.append(f"| {size} | {ft_wer:.2f}% | {orig_wer:.2f}% | {improvement:+.1f}% | {result} |")

    lines.append("\n## Hebrew Code-Switching Analysis\n")
    lines.append("| Model Size | Fine-tuned WER | Original WER | Improvement | Result |")
    lines.append("|------------|----------------|--------------|-------------|--------|")

    for size in MODEL_SIZES:
        ft_key = f'finetune-{size}'
        orig_key = f'original-{size}'

        ft_wer = hebrew_wer.get(ft_key, {}).get('_overall', {}).get('mean', 0) * 100
        orig_wer = hebrew_wer.get(orig_key, {}).get('_overall', {}).get('mean', 0) * 100

        if orig_wer > 0:
            improvement = ((orig_wer - ft_wer) / orig_wer) * 100
            result = "**Fine-tune better**" if improvement > 0 else "Fine-tune worse"
        else:
            improvement = 0
            result = "N/A"

        lines.append(f"| {size} | {ft_wer:.2f}% | {orig_wer:.2f}% | {improvement:+.1f}% | {result} |")

    return '\n'.join(lines)


def create_category_breakdown(tech_wer: dict, hebrew_wer: dict) -> str:
    """Create detailed category breakdown."""
    lines = []

    lines.append("\n## Detailed Category Breakdown\n")

    # Technical categories
    lines.append("### Technical Vocabulary by Category\n")
    for category in TECH_VOCAB_CATEGORIES:
        lines.append(f"\n#### {category.replace('_', ' ').title()}\n")
        lines.append("| Model Size | Fine-tuned | Original | Diff |")
        lines.append("|------------|------------|----------|------|")

        for size in MODEL_SIZES:
            ft_key = f'finetune-{size}'
            orig_key = f'original-{size}'

            ft_data = tech_wer.get(ft_key, {}).get(category, {})
            orig_data = tech_wer.get(orig_key, {}).get(category, {})

            ft_mean = ft_data.get('mean', 0) * 100
            orig_mean = orig_data.get('mean', 0) * 100
            diff = ft_mean - orig_mean

            lines.append(f"| {size} | {ft_mean:.2f}% | {orig_mean:.2f}% | {diff:+.2f}% |")

    # Hebrew categories
    lines.append("\n### Hebrew Code-Switching by Category\n")
    for category in HEBREW_CODESWITCHING_CATEGORIES:
        lines.append(f"\n#### {category.replace('_', ' ').title()}\n")
        lines.append("| Model Size | Fine-tuned | Original | Diff |")
        lines.append("|------------|------------|----------|------|")

        for size in MODEL_SIZES:
            ft_key = f'finetune-{size}'
            orig_key = f'original-{size}'

            ft_data = hebrew_wer.get(ft_key, {}).get(category, {})
            orig_data = hebrew_wer.get(orig_key, {}).get(category, {})

            ft_mean = ft_data.get('mean', 0) * 100
            orig_mean = orig_data.get('mean', 0) * 100
            diff = ft_mean - orig_mean

            lines.append(f"| {size} | {ft_mean:.2f}% | {orig_mean:.2f}% | {diff:+.2f}% |")

    return '\n'.join(lines)


def create_visualizations(tech_wer: dict, hebrew_wer: dict):
    """Create comparison visualizations."""
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')

    # 1. Technical Vocabulary Comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(MODEL_SIZES))
    width = 0.35

    ft_wers = [tech_wer.get(f'finetune-{s}', {}).get('_overall', {}).get('mean', 0) * 100 for s in MODEL_SIZES]
    orig_wers = [tech_wer.get(f'original-{s}', {}).get('_overall', {}).get('mean', 0) * 100 for s in MODEL_SIZES]

    bars1 = ax.bar(x - width/2, ft_wers, width, label='Fine-tuned', color='#2ecc71')
    bars2 = ax.bar(x + width/2, orig_wers, width, label='Original', color='#3498db')

    ax.set_xlabel('Model Size', fontsize=12)
    ax.set_ylabel('Word Error Rate (%)', fontsize=12)
    ax.set_title('Technical Vocabulary: Fine-tuned vs Original WER', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in MODEL_SIZES])
    ax.legend()
    ax.bar_label(bars1, fmt='%.2f%%', padding=3, fontsize=8)
    ax.bar_label(bars2, fmt='%.2f%%', padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'tech_vocab_comparison.png', dpi=150)
    plt.close()

    # 2. Hebrew Code-Switching Comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    ft_wers = [hebrew_wer.get(f'finetune-{s}', {}).get('_overall', {}).get('mean', 0) * 100 for s in MODEL_SIZES]
    orig_wers = [hebrew_wer.get(f'original-{s}', {}).get('_overall', {}).get('mean', 0) * 100 for s in MODEL_SIZES]

    bars1 = ax.bar(x - width/2, ft_wers, width, label='Fine-tuned', color='#e74c3c')
    bars2 = ax.bar(x + width/2, orig_wers, width, label='Original', color='#9b59b6')

    ax.set_xlabel('Model Size', fontsize=12)
    ax.set_ylabel('Word Error Rate (%)', fontsize=12)
    ax.set_title('Hebrew Code-Switching: Fine-tuned vs Original WER', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in MODEL_SIZES])
    ax.legend()
    ax.bar_label(bars1, fmt='%.2f%%', padding=3, fontsize=8)
    ax.bar_label(bars2, fmt='%.2f%%', padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'hebrew_codeswitching_comparison.png', dpi=150)
    plt.close()

    # 3. Improvement Heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Tech vocab improvement by category and model
    tech_categories = [c for c in TECH_VOCAB_CATEGORIES if any(tech_wer.get(f'finetune-{s}', {}).get(c) for s in MODEL_SIZES)]
    tech_improvement = np.zeros((len(MODEL_SIZES), len(tech_categories)))

    for i, size in enumerate(MODEL_SIZES):
        for j, cat in enumerate(tech_categories):
            ft = tech_wer.get(f'finetune-{size}', {}).get(cat, {}).get('mean', 0)
            orig = tech_wer.get(f'original-{size}', {}).get(cat, {}).get('mean', 0)
            if orig > 0:
                tech_improvement[i, j] = ((orig - ft) / orig) * 100

    im1 = ax1.imshow(tech_improvement, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)
    ax1.set_xticks(range(len(tech_categories)))
    ax1.set_xticklabels([c.replace('tech_', '').replace('_', '\n') for c in tech_categories], rotation=45, ha='right')
    ax1.set_yticks(range(len(MODEL_SIZES)))
    ax1.set_yticklabels([s.capitalize() for s in MODEL_SIZES])
    ax1.set_title('Technical Vocab: Fine-tuning Improvement (%)', fontsize=12)

    for i in range(len(MODEL_SIZES)):
        for j in range(len(tech_categories)):
            val = tech_improvement[i, j]
            color = 'white' if abs(val) > 25 else 'black'
            ax1.text(j, i, f'{val:.1f}%', ha='center', va='center', color=color, fontsize=8)

    plt.colorbar(im1, ax=ax1, label='Improvement %')

    # Hebrew codeswitching improvement
    hebrew_categories = HEBREW_CODESWITCHING_CATEGORIES
    hebrew_improvement = np.zeros((len(MODEL_SIZES), len(hebrew_categories)))

    for i, size in enumerate(MODEL_SIZES):
        for j, cat in enumerate(hebrew_categories):
            ft = hebrew_wer.get(f'finetune-{size}', {}).get(cat, {}).get('mean', 0)
            orig = hebrew_wer.get(f'original-{size}', {}).get(cat, {}).get('mean', 0)
            if orig > 0:
                hebrew_improvement[i, j] = ((orig - ft) / orig) * 100

    im2 = ax2.imshow(hebrew_improvement, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)
    ax2.set_xticks(range(len(hebrew_categories)))
    ax2.set_xticklabels([c.replace('hebrew_', '').replace('_', '\n') for c in hebrew_categories])
    ax2.set_yticks(range(len(MODEL_SIZES)))
    ax2.set_yticklabels([s.capitalize() for s in MODEL_SIZES])
    ax2.set_title('Hebrew Code-Switching: Fine-tuning Improvement (%)', fontsize=12)

    for i in range(len(MODEL_SIZES)):
        for j in range(len(hebrew_categories)):
            val = hebrew_improvement[i, j]
            color = 'white' if abs(val) > 25 else 'black'
            ax2.text(j, i, f'{val:.1f}%', ha='center', va='center', color=color, fontsize=8)

    plt.colorbar(im2, ax=ax2, label='Improvement %')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'improvement_heatmap.png', dpi=150)
    plt.close()

    # 4. Combined comparison chart
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(MODEL_SIZES))
    width = 0.2

    # Calculate improvements
    tech_improvements = []
    hebrew_improvements = []
    overall_improvements = []

    for size in MODEL_SIZES:
        ft_tech = tech_wer.get(f'finetune-{size}', {}).get('_overall', {}).get('mean', 0)
        orig_tech = tech_wer.get(f'original-{size}', {}).get('_overall', {}).get('mean', 0)

        ft_hebrew = hebrew_wer.get(f'finetune-{size}', {}).get('_overall', {}).get('mean', 0)
        orig_hebrew = hebrew_wer.get(f'original-{size}', {}).get('_overall', {}).get('mean', 0)

        tech_imp = ((orig_tech - ft_tech) / orig_tech * 100) if orig_tech > 0 else 0
        hebrew_imp = ((orig_hebrew - ft_hebrew) / orig_hebrew * 100) if orig_hebrew > 0 else 0

        tech_improvements.append(tech_imp)
        hebrew_improvements.append(hebrew_imp)
        overall_improvements.append((tech_imp + hebrew_imp) / 2)

    bars1 = ax.bar(x - width, tech_improvements, width, label='Technical Vocab', color='#3498db')
    bars2 = ax.bar(x, hebrew_improvements, width, label='Hebrew Code-Switching', color='#e74c3c')
    bars3 = ax.bar(x + width, overall_improvements, width, label='Average', color='#2ecc71')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Model Size', fontsize=12)
    ax.set_ylabel('WER Improvement (%)', fontsize=12)
    ax.set_title('Fine-Tuning Impact: WER Improvement by Category\n(Positive = Fine-tune Better)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in MODEL_SIZES])
    ax.legend()

    for bars in [bars1, bars2, bars3]:
        ax.bar_label(bars, fmt='%+.1f%%', padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'subset_improvement_comparison.png', dpi=150)
    plt.close()

    print(f"Visualizations saved to {VIZ_DIR}")


def main():
    print("Loading results...")
    results = load_results()

    print("Calculating WER for technical vocabulary...")
    tech_wer = calculate_category_wer(results, TECH_VOCAB_CATEGORIES)

    print("Calculating WER for Hebrew code-switching...")
    hebrew_wer = calculate_category_wer(results, HEBREW_CODESWITCHING_CATEGORIES)

    # Create analysis directory
    ANALYSIS_DIR.mkdir(exist_ok=True)

    # Generate report
    print("Generating analysis report...")
    report = create_comparison_table(tech_wer, hebrew_wer)
    report += create_category_breakdown(tech_wer, hebrew_wer)

    # Add key findings
    report += "\n\n## Key Findings\n"

    # Calculate summary stats
    tech_improved = 0
    hebrew_improved = 0

    for size in MODEL_SIZES:
        ft_tech = tech_wer.get(f'finetune-{size}', {}).get('_overall', {}).get('mean', 0)
        orig_tech = tech_wer.get(f'original-{size}', {}).get('_overall', {}).get('mean', 0)
        if ft_tech < orig_tech:
            tech_improved += 1

        ft_hebrew = hebrew_wer.get(f'finetune-{size}', {}).get('_overall', {}).get('mean', 0)
        orig_hebrew = hebrew_wer.get(f'original-{size}', {}).get('_overall', {}).get('mean', 0)
        if ft_hebrew < orig_hebrew:
            hebrew_improved += 1

    report += f"\n- **Technical Vocabulary**: Fine-tuning improved {tech_improved}/5 model sizes\n"
    report += f"- **Hebrew Code-Switching**: Fine-tuning improved {hebrew_improved}/5 model sizes\n"

    # Find best performing models
    best_tech_ft = min(MODEL_SIZES, key=lambda s: tech_wer.get(f'finetune-{s}', {}).get('_overall', {}).get('mean', 1))
    best_hebrew_ft = min(MODEL_SIZES, key=lambda s: hebrew_wer.get(f'finetune-{s}', {}).get('_overall', {}).get('mean', 1))

    report += f"\n### Best Performing Fine-tuned Models:\n"
    report += f"- **Technical Vocabulary**: {best_tech_ft} ({tech_wer[f'finetune-{best_tech_ft}']['_overall']['mean']*100:.2f}% WER)\n"
    report += f"- **Hebrew Code-Switching**: {best_hebrew_ft} ({hebrew_wer[f'finetune-{best_hebrew_ft}']['_overall']['mean']*100:.2f}% WER)\n"

    # Write report
    report_path = ANALYSIS_DIR / 'subset_analysis.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(tech_wer, hebrew_wer)

    # Also save raw data as JSON
    analysis_data = {
        'tech_vocab': {k: {cat: {sk: sv for sk, sv in v.items() if sk != 'values'}
                          for cat, v in cats.items()}
                      for k, cats in tech_wer.items()},
        'hebrew_codeswitching': {k: {cat: {sk: sv for sk, sv in v.items() if sk != 'values'}
                                    for cat, v in cats.items()}
                                for k, cats in hebrew_wer.items()},
    }

    json_path = ANALYSIS_DIR / 'subset_analysis_data.json'
    with open(json_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    print(f"Data saved to {json_path}")

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Generate comparison visualizations for fine-tuned vs original Whisper models."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data from evaluation results
models = ['tiny', 'base', 'small', 'medium', 'large-v3-turbo']
models_short = ['tiny', 'base', 'small', 'medium', 'large']

# WER percentages
finetune_wer = [6.59, 5.02, 3.45, 2.96, 3.01]
original_wer = [7.85, 5.46, 3.96, 2.71, 2.77]

# Inference time (seconds per file)
finetune_time = [0.247, 0.408, 0.823, 1.462, 0.991]
original_time = [0.275, 0.413, 0.518, 1.084, 1.004]

# Output directory
output_dir = Path(__file__).parent.parent / 'visualizations'
output_dir.mkdir(exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
colors = {'finetune': '#2E86AB', 'original': '#A23B72'}

def create_wer_comparison():
    """Create WER comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models_short))
    width = 0.35

    bars1 = ax.bar(x - width/2, finetune_wer, width, label='Fine-tuned', color=colors['finetune'])
    bars2 = ax.bar(x + width/2, original_wer, width, label='Original', color=colors['original'])

    ax.set_xlabel('Model Size', fontsize=12)
    ax.set_ylabel('Word Error Rate (%)', fontsize=12)
    ax.set_title('WER Comparison: Fine-tuned vs Original Whisper Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_short)
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(max(finetune_wer), max(original_wer)) * 1.2)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    # Add improvement annotations
    for i, (ft, orig) in enumerate(zip(finetune_wer, original_wer)):
        improvement = ((orig - ft) / orig) * 100
        if improvement > 0:
            color = 'green'
            symbol = '+'
        else:
            color = 'red'
            symbol = ''
        ax.annotate(f'{symbol}{improvement:.0f}%',
                    xy=(i, max(ft, orig) + 0.8),
                    ha='center', va='bottom', fontsize=10,
                    fontweight='bold', color=color)

    # Add highlight box over medium and large models (indices 3 and 4)
    from matplotlib.patches import FancyBboxPatch

    # Calculate box position (spanning medium and large)
    box_left = 3 - width - 0.15
    box_right = 4 + width + 0.15
    box_width = box_right - box_left
    box_height = max(finetune_wer[3:5] + original_wer[3:5]) + 1.8

    # Add semi-transparent highlight box
    highlight_box = FancyBboxPatch(
        (box_left, -0.1), box_width, box_height,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        facecolor='#FFCCCC', edgecolor='#CC0000',
        alpha=0.25, linewidth=2, linestyle='--',
        zorder=0
    )
    ax.add_patch(highlight_box)

    # Add annotation text above the highlight box
    ax.annotate(
        'Fine-tuned models\nperformed worse',
        xy=((3 + 4) / 2, box_height + 0.3),
        ha='center', va='bottom', fontsize=10,
        fontweight='bold', color='#990000',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEEEE',
                  edgecolor='#CC0000', linewidth=1.5)
    )

    plt.tight_layout()
    plt.savefig(output_dir / 'wer_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'wer_comparison.png'}")

def create_inference_time_comparison():
    """Create inference time comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models_short))
    width = 0.35

    bars1 = ax.bar(x - width/2, finetune_time, width, label='Fine-tuned', color=colors['finetune'])
    bars2 = ax.bar(x + width/2, original_time, width, label='Original', color=colors['original'])

    ax.set_xlabel('Model Size', fontsize=12)
    ax.set_ylabel('Inference Time (seconds per file)', fontsize=12)
    ax.set_title('Inference Time: Fine-tuned vs Original Whisper Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_short)
    ax.legend(loc='upper left')
    ax.set_ylim(0, max(max(finetune_time), max(original_time)) * 1.2)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'inference_time_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'inference_time_comparison.png'}")

def create_improvement_chart():
    """Create chart showing WER improvement/degradation from fine-tuning."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate percentage improvement (positive = better, negative = worse)
    improvements = [((orig - ft) / orig) * 100 for ft, orig in zip(finetune_wer, original_wer)]

    x = np.arange(len(models_short))

    # Color bars based on improvement (green) or degradation (red)
    bar_colors = ['#2ECC71' if imp > 0 else '#E74C3C' for imp in improvements]

    bars = ax.bar(x, improvements, color=bar_colors, edgecolor='black', linewidth=0.5)

    # Add zero baseline
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)

    ax.set_xlabel('Model Size', fontsize=12)
    ax.set_ylabel('WER Improvement (%)', fontsize=12)
    ax.set_title('Fine-Tuning Impact on Word Error Rate\n(Positive = Improvement, Negative = Degradation)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_short)

    # Symmetric y-axis for visual balance
    max_abs = max(abs(min(improvements)), abs(max(improvements)))
    ax.set_ylim(-max_abs * 1.3, max_abs * 1.3)

    # Add value labels on bars
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 3 if height >= 0 else -3
        label = f'+{imp:.1f}%' if imp > 0 else f'{imp:.1f}%'
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset), textcoords="offset points",
                    ha='center', va=va, fontsize=11, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ECC71', edgecolor='black', label='Improvement'),
                       Patch(facecolor='#E74C3C', edgecolor='black', label='Degradation')]
    ax.legend(handles=legend_elements, loc='upper right')

    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'finetune_improvement.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'finetune_improvement.png'}")

def create_combined_chart():
    """Create a combined chart showing both WER and inference time."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(models_short))
    width = 0.35

    # WER chart
    bars1 = ax1.bar(x - width/2, finetune_wer, width, label='Fine-tuned', color=colors['finetune'])
    bars2 = ax1.bar(x + width/2, original_wer, width, label='Original', color=colors['original'])
    ax1.set_xlabel('Model Size', fontsize=11)
    ax1.set_ylabel('Word Error Rate (%)', fontsize=11)
    ax1.set_title('Word Error Rate', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_short)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 10)

    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    # Inference time chart
    bars3 = ax2.bar(x - width/2, finetune_time, width, label='Fine-tuned', color=colors['finetune'])
    bars4 = ax2.bar(x + width/2, original_time, width, label='Original', color=colors['original'])
    ax2.set_xlabel('Model Size', fontsize=11)
    ax2.set_ylabel('Seconds per file', fontsize=11)
    ax2.set_title('Inference Time', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models_short)
    ax2.legend(loc='upper left')
    ax2.set_ylim(0, 1.8)

    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}s', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}s', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.suptitle('Fine-tuned vs Original Whisper Models', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'combined_comparison.png'}")

if __name__ == '__main__':
    create_wer_comparison()
    create_inference_time_comparison()
    create_improvement_chart()
    create_combined_chart()
    print("\nAll visualizations created successfully!")

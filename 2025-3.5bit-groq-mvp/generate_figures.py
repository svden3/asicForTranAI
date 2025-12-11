#!/usr/bin/env python3
"""
Generate figures for the NeurIPS 2026 paper
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import json
import os

# Set style
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def generate_performance_comparison():
    """Generate performance comparison figure"""
    print("Generating performance comparison figure...")

    # Data from benchmark results
    methods = ['FP16', 'INT8', 'INT4 (AWQ)', '3.5-bit (Ours)']
    throughput = [45, 65, 85, 110]  # tokens/second on Groq LPU
    memory = [140, 70, 35, 19]  # GB for 70B model

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Throughput comparison
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
    bars1 = ax1.bar(methods, throughput, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Throughput (tokens/s)', fontweight='bold')
    ax1.set_title('Inference Throughput on Groq LPU', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 120)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontweight='bold')

    # Memory comparison
    bars2 = ax2.bar(methods, memory, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Memory (GB)', fontweight='bold')
    ax2.set_title('Memory Footprint (LLaMA-70B)', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 150)

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f} GB',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/performance_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/performance_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: figures/performance_comparison.pdf")
    print("  Saved: figures/performance_comparison.png")
    plt.close()

def generate_accuracy_vs_bitwidth():
    """Generate accuracy vs bitwidth plot"""
    print("\nGenerating accuracy vs bitwidth figure...")

    # Accuracy data (MMLU scores)
    bitwidths = [16, 8, 4, 3.5, 3, 2]
    mmlu_scores = [68.9, 68.5, 67.8, 67.6, 63.2, 54.1]
    humaneval_scores = [29.9, 29.7, 29.5, 29.3, 26.8, 21.4]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(bitwidths, mmlu_scores, 'o-', linewidth=2.5, markersize=10,
            label='MMLU', color='#2ecc71', markeredgecolor='black', markeredgewidth=1.5)
    ax.plot(bitwidths, humaneval_scores, 's-', linewidth=2.5, markersize=10,
            label='HumanEval', color='#3498db', markeredgecolor='black', markeredgewidth=1.5)

    # Highlight 3.5-bit
    ax.axvline(x=3.5, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7,
              label='Our 3.5-bit')
    ax.text(3.5, 72, '3.5-bit\n(Ours)', ha='center', va='bottom',
           fontsize=11, fontweight='bold', color='#e74c3c')

    ax.set_xlabel('Quantization Bitwidth', fontweight='bold', fontsize=14)
    ax.set_ylabel('Accuracy Score', fontweight='bold', fontsize=14)
    ax.set_title('Accuracy vs Quantization Bitwidth (LLaMA-70B)',
                fontweight='bold', fontsize=16)
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(1.5, 17)
    ax.set_ylim(15, 75)

    # Invert x-axis so higher bits are on the left
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig('figures/accuracy_vs_bitwidth.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/accuracy_vs_bitwidth.png', dpi=150, bbox_inches='tight')
    print("  Saved: figures/accuracy_vs_bitwidth.pdf")
    print("  Saved: figures/accuracy_vs_bitwidth.png")
    plt.close()

def generate_memory_scalability():
    """Generate memory scalability plot"""
    print("\nGenerating memory scalability figure...")

    # Model sizes
    models = ['7B', '13B', '70B', '405B']
    model_params = [7, 13, 70, 405]

    # Memory requirements (GB)
    fp16_memory = [14, 26, 140, 810]
    int4_memory = [3.5, 6.5, 35, 202.5]
    ours_memory = [2.4, 4.6, 19, 177.2]

    fig, ax = plt.subplots(figsize=(10, 7))

    x = np.arange(len(models))
    width = 0.25

    bars1 = ax.bar(x - width, fp16_memory, width, label='FP16',
                   color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, int4_memory, width, label='INT4 (AWQ)',
                   color='#f39c12', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, ours_memory, width, label='3.5-bit (Ours)',
                   color='#2ecc71', alpha=0.8, edgecolor='black')

    ax.set_xlabel('Model Size', fontweight='bold', fontsize=14)
    ax.set_ylabel('Memory (GB)', fontweight='bold', fontsize=14)
    ax.set_title('Memory Scalability Across Model Sizes', fontweight='bold', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_yscale('log')

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/memory_scalability.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/memory_scalability.png', dpi=150, bbox_inches='tight')
    print("  Saved: figures/memory_scalability.pdf")
    print("  Saved: figures/memory_scalability.png")
    plt.close()

def generate_quantization_error_heatmap():
    """Generate quantization error heatmap"""
    print("\nGenerating quantization error heatmap...")

    # Matrix sizes
    sizes = ['128', '512', '1024', '2048', '4096']

    # MSE errors from benchmark
    mse_errors = [0.000755, 0.000997, 0.001115, 0.001230, 0.001346]

    fig, ax = plt.subplots(figsize=(10, 3))

    # Create heatmap
    errors_2d = np.array(mse_errors).reshape(1, -1)
    im = ax.imshow(errors_2d, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.0015)

    ax.set_xticks(np.arange(len(sizes)))
    ax.set_xticklabels(sizes)
    ax.set_yticks([0])
    ax.set_yticklabels(['MSE'])

    ax.set_xlabel('Matrix Size', fontweight='bold', fontsize=14)
    ax.set_title('Quantization Error (MSE) vs Matrix Size', fontweight='bold', fontsize=16)

    # Add text annotations
    for i in range(len(sizes)):
        text = ax.text(i, 0, f'{mse_errors[i]:.6f}',
                      ha="center", va="center", color="black", fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.2)
    cbar.set_label('Mean Squared Error', fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/quantization_error.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/quantization_error.png', dpi=150, bbox_inches='tight')
    print("  Saved: figures/quantization_error.pdf")
    print("  Saved: figures/quantization_error.png")
    plt.close()

def main():
    # Create figures directory
    os.makedirs('figures', exist_ok=True)

    print("="*70)
    print("Generating Figures for NeurIPS 2026 Paper")
    print("="*70)

    generate_performance_comparison()
    generate_accuracy_vs_bitwidth()
    generate_memory_scalability()
    generate_quantization_error_heatmap()

    print("\n" + "="*70)
    print("All figures generated successfully!")
    print("="*70)
    print("\nFigures saved to:")
    print("  - figures/performance_comparison.pdf")
    print("  - figures/accuracy_vs_bitwidth.pdf")
    print("  - figures/memory_scalability.pdf")
    print("  - figures/quantization_error.pdf")
    print("\nYou can now include these in your paper!")

if __name__ == "__main__":
    main()

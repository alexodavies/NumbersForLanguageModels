#!/usr/bin/env python3
"""
Visualization script for model performance metrics across different number formats and sizes.
Each format is shown as a separate line in the plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Define format styles and colors
FORMAT_STYLES = {
    'standard': {'marker': 'o', 'color': '#2E86AB', 'linestyle': '-', 'label': 'Standard'},
    'scientific_3sf': {'marker': 's', 'color': '#A23B72', 'linestyle': '--', 'label': 'Scientific (3SF)'},
    'rounded_3sf': {'marker': '^', 'color': '#F18F01', 'linestyle': '-.', 'label': 'Rounded (3SF)'},
    'written_words': {'marker': 'd', 'color': '#BC4749', 'linestyle': ':', 'label': 'Written Words'}
}

def load_data(filepath):
    """Load and validate the CSV data."""
    data = pd.read_csv(filepath)
    print(f"Loaded {len(data)} rows of data")
    print(f"Models: {data['model'].unique()}")
    print(f"Formats: {data['format'].unique()}")
    print(f"Size range: {data['size'].min()} to {data['size'].max()}")
    return data

def create_model_plots_by_format(data, output_dir):
    """Create plots for each model showing different formats as separate lines."""
    models = data['model'].unique()
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    for model in models:
        # Filter data for this model
        model_data = data[data['model'] == model]
        formats = model_data['format'].unique()
        
        # Create figure with 3 subplots (3 rows, 1 column)
        fig, axes = plt.subplots(3, 1, figsize=(12, 14))
        fig.suptitle(f'Model: {model}\nPerformance Metrics by Number Format', 
                    fontsize=16, fontweight='bold')
        
        # Plot each format as a separate line
        for fmt in formats:
            fmt_data = model_data[model_data['format'] == fmt].sort_values('size')
            style = FORMAT_STYLES.get(fmt, {'marker': 'o', 'color': 'gray', 
                                           'linestyle': '-', 'label': fmt})
            
            # Plot 1: Linear R²
            axes[0].errorbar(fmt_data['size'], 
                           fmt_data['linear_r2_mean'], 
                           yerr=fmt_data['linear_r2_std'],
                           linestyle=style['linestyle'],
                           capsize=4,
                           capthick=1.5,
                           linewidth=2,
                           markersize=7,
                           label=style['label'],
                           alpha=0.9)
            
            # Plot 2: PCA R²
            axes[1].errorbar(fmt_data['size'], 
                           fmt_data['pca_r2_mean'], 
                           yerr=fmt_data['pca_r2_std'],
                           linestyle=style['linestyle'],
                           capsize=4,
                           capthick=1.5,
                           linewidth=2,
                           markersize=7,
                           label=style['label'],
                           alpha=0.9)
            
            # Plot 3: First Component Explained Variance
            axes[2].errorbar(fmt_data['size'], 
                           fmt_data['pca_var_comp_1_mean'], 
                           yerr=fmt_data['pca_var_comp_1_std'],
                           linestyle=style['linestyle'],
                           capsize=4,
                           capthick=1.5,
                           linewidth=2,
                           markersize=7,
                           label=style['label'],
                           alpha=0.9)
        
        # Configure subplot 1 (Linear R²)
        axes[0].set_xlabel('Number Size', fontsize=12)
        axes[0].set_ylabel('Linear R² Score', fontsize=12)
        axes[0].set_title('Linear Regression R² vs Number Size', fontsize=14, pad=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='best', fontsize=10, framealpha=0.95)
        axes[0].set_ylim([-0.1, 1.05])
        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Configure subplot 2 (PCA R²)
        axes[1].set_xlabel('Number Size', fontsize=12)
        axes[1].set_ylabel('PCA R² Score', fontsize=12)
        axes[1].set_title('PCA R² vs Number Size', fontsize=14, pad=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='best', fontsize=10, framealpha=0.95)
        axes[1].set_ylim([-0.1, 1.05])
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Configure subplot 3 (First Component Variance)
        axes[2].set_xlabel('Number Size', fontsize=12)
        axes[2].set_ylabel('Explained Variance Ratio', fontsize=12)
        axes[2].set_title('First PCA Component Explained Variance vs Number Size', fontsize=14, pad=10)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc='best', fontsize=10, framealpha=0.95)
        axes[2].set_ylim([0, max(model_data['pca_var_comp_1_mean'].max() * 1.1, 0.25)])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_filename = os.path.join(output_dir, 
                                      f'{model.replace("/", "_")}_format_comparison.png')
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"Saved figure: {output_filename}")
        plt.close()

def create_format_comparison_grid(data, output_dir):
    """Create a grid of plots comparing formats side by side."""
    models = data['model'].unique()
    
    for model in models:
        model_data = data[data['model'] == model]
        formats = sorted(model_data['format'].unique())
        
        # Create a 3x4 grid (3 metrics x 4 formats)
        fig, axes = plt.subplots(3, len(formats), figsize=(16, 10))
        if len(formats) == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'Model: {model} - Format Comparison Grid', 
                    fontsize=16, fontweight='bold')
        
        metrics = [
            ('linear_r2_mean', 'linear_r2_std', 'Linear R²'),
            ('pca_r2_mean', 'pca_r2_std', 'PCA R²'),
            ('pca_var_comp_1_mean', 'pca_var_comp_1_std', '1st Comp. Variance')
        ]
        
        for col_idx, fmt in enumerate(formats):
            fmt_data = model_data[model_data['format'] == fmt].sort_values('size')
            style = FORMAT_STYLES.get(fmt, {'color': 'gray', 'label': fmt})
            
            for row_idx, (mean_col, std_col, metric_name) in enumerate(metrics):
                ax = axes[row_idx, col_idx]
                
                ax.errorbar(fmt_data['size'], 
                          fmt_data[mean_col], 
                          yerr=fmt_data[std_col],
                          marker='o',
                          color=style['color'],
                          capsize=3,
                          linewidth=2,
                          markersize=6,
                          alpha=0.9)
                
                # Add trend line
                if len(fmt_data) > 2:
                    z = np.polyfit(fmt_data['size'], fmt_data[mean_col], 2)
                    p = np.poly1d(z)
                    x_smooth = np.linspace(fmt_data['size'].min(), fmt_data['size'].max(), 100)
                    ax.plot(x_smooth, p(x_smooth), '--', alpha=0.3, color=style['color'])
                
                if row_idx == 0:
                    ax.set_title(style['label'], fontsize=12, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel(metric_name, fontsize=10)
                if row_idx == len(metrics) - 1:
                    ax.set_xlabel('Size', fontsize=10)
                
                ax.grid(True, alpha=0.2)
                ax.tick_params(labelsize=8)
                
                # Set y-limits based on metric type
                if 'r2' in mean_col.lower():
                    ax.set_ylim([-0.1, 1.05])
                    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_filename = os.path.join(output_dir, 
                                      f'{model.replace("/", "_")}_format_grid.png')
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"Saved grid figure: {output_filename}")
        plt.close()

def create_summary_comparison(data, output_dir):
    """Create a summary plot showing key differences between formats."""
    models = data['model'].unique()
    
    for model in models:
        model_data = data[data['model'] == model]
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Model: {model} - Format Performance Summary', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Average Linear R² across all sizes
        ax1 = axes[0, 0]
        for fmt in model_data['format'].unique():
            fmt_data = model_data[model_data['format'] == fmt]
            style = FORMAT_STYLES.get(fmt, {'color': 'gray', 'label': fmt})
            
            # Calculate mean across all sizes
            mean_by_size = fmt_data.groupby('size')['linear_r2_mean'].mean()
            ax1.bar(fmt, mean_by_size.mean(), color=style['color'], 
                   label=style['label'], alpha=0.7)
        
        ax1.set_ylabel('Average Linear R²', fontsize=11)
        ax1.set_title('Average Linear R² (across all sizes)', fontsize=12)
        ax1.set_ylim([0, 1])
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Performance degradation (size 1 to max size)
        ax2 = axes[0, 1]
        for fmt in model_data['format'].unique():
            fmt_data = model_data[model_data['format'] == fmt].sort_values('size')
            if len(fmt_data) > 0:
                style = FORMAT_STYLES.get(fmt, {'color': 'gray', 'label': fmt})
                
                first_val = fmt_data.iloc[0]['linear_r2_mean']
                last_val = fmt_data.iloc[-1]['linear_r2_mean']
                degradation = first_val - last_val
                
                ax2.bar(style['label'], degradation, color=style['color'], alpha=0.7)
        
        ax2.set_ylabel('R² Degradation', fontsize=11)
        ax2.set_title('Linear R² Degradation (First to Last Size)', fontsize=12)
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: PCA R² stability (standard deviation of means)
        ax3 = axes[1, 0]
        for fmt in model_data['format'].unique():
            fmt_data = model_data[model_data['format'] == fmt]
            style = FORMAT_STYLES.get(fmt, {'color': 'gray', 'label': fmt})
            
            # Calculate stability as std of means across sizes
            stability = fmt_data['pca_r2_mean'].std()
            ax3.bar(style['label'], stability, color=style['color'], alpha=0.7)
        
        ax3.set_ylabel('Std Dev of PCA R²', fontsize=11)
        ax3.set_title('PCA R² Variability Across Sizes', fontsize=12)
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: First component variance at size=10
        ax4 = axes[1, 1]
        for fmt in model_data['format'].unique():
            fmt_data = model_data[(model_data['format'] == fmt) & (model_data['size'] == 10)]
            if len(fmt_data) > 0:
                style = FORMAT_STYLES.get(fmt, {'color': 'gray', 'label': fmt})
                ax4.bar(style['label'], fmt_data.iloc[0]['pca_var_comp_1_mean'], 
                       color=style['color'], alpha=0.7)
        
        ax4.set_ylabel('1st Component Variance', fontsize=11)
        ax4.set_title('First PCA Component Variance at Size=10', fontsize=12)
        ax4.grid(axis='y', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_filename = os.path.join(output_dir, 
                                      f'{model.replace("/", "_")}_format_summary.png')
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"Saved summary figure: {output_filename}")
        plt.close()

def print_detailed_statistics(data):
    """Print detailed statistics for each format."""
    print("\n" + "="*80)
    print("DETAILED STATISTICS BY FORMAT")
    print("="*80)
    
    models = data['model'].unique()
    
    for model in models:
        print(f"\nModel: {model}")
        print("-"*60)
        
        model_data = data[data['model'] == model]
        
        for fmt in sorted(model_data['format'].unique()):
            fmt_data = model_data[model_data['format'] == fmt]
            style = FORMAT_STYLES.get(fmt, {'label': fmt})
            
            print(f"\n  Format: {style['label']}")
            print("  " + "-"*40)
            
            # Linear R²
            print(f"    Linear R²:")
            print(f"      Range: {fmt_data['linear_r2_mean'].min():.4f} to {fmt_data['linear_r2_mean'].max():.4f}")
            print(f"      Mean: {fmt_data['linear_r2_mean'].mean():.4f}")
            print(f"      Degradation: {fmt_data['linear_r2_mean'].max() - fmt_data['linear_r2_mean'].min():.4f}")
            
            # PCA R²
            print(f"    PCA R²:")
            print(f"      Range: {fmt_data['pca_r2_mean'].min():.4f} to {fmt_data['pca_r2_mean'].max():.4f}")
            print(f"      Mean: {fmt_data['pca_r2_mean'].mean():.4f}")
            print(f"      Std Dev: {fmt_data['pca_r2_mean'].std():.4f}")
            
            # First Component Variance
            print(f"    1st Component Variance:")
            print(f"      Range: {fmt_data['pca_var_comp_1_mean'].min():.4f} to {fmt_data['pca_var_comp_1_mean'].max():.4f}")
            print(f"      Mean: {fmt_data['pca_var_comp_1_mean'].mean():.4f}")
    
    print("\n" + "="*80)

def main():
    """Main function to run the visualization pipeline."""
    # Set paths
    csv_path = 'results/format_size_sweep/number_size_sweep_results.csv'
    output_dir = 'final_plots/formats'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {csv_path}")
    data = load_data(csv_path)
    
    # Create visualizations
    print("\n" + "="*60)
    print("Creating visualizations...")
    print("="*60)
    
    print("\n1. Creating format comparison plots (lines for each format)...")
    create_model_plots_by_format(data, output_dir)
    
    print("\n2. Creating format comparison grid...")
    create_format_comparison_grid(data, output_dir)
    
    print("\n3. Creating summary comparison plots...")
    create_summary_comparison(data, output_dir)
    
    # Print statistics
    print_detailed_statistics(data)
    
    print(f"\n✅ All visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - *_format_comparison.png: Main plots with different formats as lines")
    print("  - *_format_grid.png: Grid view of each format separately")
    print("  - *_format_summary.png: Summary statistics comparison")
    print("\nDone!")

if __name__ == "__main__":
    main()
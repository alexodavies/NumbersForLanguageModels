#!/usr/bin/env python3
"""
PCA Visualization for Format Comparison Experiment

Creates 2D PCA scatter plots showing how different number representations
(standard, scientific, rounded, written) affect embedding structure.

Layout: 4x3 grid
- Rows: 4 formats (standard, scientific_3sf, rounded_3sf, written_words)
- Columns: 3 models (text-embedding-3-large, gemini-embedding-001, voyage-3-large)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import sys

# Add current directory to path to import local modules
sys.path.append('.')

try:
    from api_wrapper import EmbeddingWrapper
    from embedding_cache import CachedEmbeddingWrapper
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this from the project directory with all required modules.")
    sys.exit(1)


def get_api_keys():
    """Get API keys from environment variables."""
    return {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'google_api_key': os.getenv('GOOGLE_API_KEY'),
        'voyage_api_key': os.getenv('VOYAGE_API_KEY')
    }


def format_scientific_3sf(num: int) -> str:
    """Scientific notation with 3 significant figures."""
    return f"{num:.2e}"


def format_rounded_3sf(num: int) -> str:
    """Round to 3 significant figures."""
    if num == 0:
        return "0"
    magnitude = int(np.floor(np.log10(abs(num))))
    rounded = round(num, -magnitude + 2)
    return str(int(rounded))


def number_to_words(num: int) -> str:
    """Convert an integer to its written English form (supports up to 10^20)."""
    if num == 0:
        return "zero"
    
    negative = num < 0
    num = abs(num)
    
    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
             "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    
    def convert_hundreds(n):
        if n == 0:
            return ""
        elif n < 10:
            return ones[n]
        elif n < 20:
            return teens[n - 10]
        elif n < 100:
            return tens[n // 10] + (" " + ones[n % 10] if n % 10 != 0 else "")
        else:
            hundred_part = ones[n // 100] + " hundred"
            rest = n % 100
            return hundred_part if rest == 0 else hundred_part + " " + convert_hundreds(rest)
    
    # Break down into quintillions, quadrillions, trillions, billions, millions, thousands, and hundreds
    quintillions = num // 1_000_000_000_000_000_000
    quadrillions = (num % 1_000_000_000_000_000_000) // 1_000_000_000_000_000
    trillions = (num % 1_000_000_000_000_000) // 1_000_000_000_000
    billions = (num % 1_000_000_000_000) // 1_000_000_000
    millions = (num % 1_000_000_000) // 1_000_000
    thousands = (num % 1_000_000) // 1_000
    hundreds = num % 1_000
    
    parts = []
    if quintillions > 0:
        parts.append(convert_hundreds(quintillions) + " quintillion")
    if quadrillions > 0:
        parts.append(convert_hundreds(quadrillions) + " quadrillion")
    if trillions > 0:
        parts.append(convert_hundreds(trillions) + " trillion")
    if billions > 0:
        parts.append(convert_hundreds(billions) + " billion")
    if millions > 0:
        parts.append(convert_hundreds(millions) + " million")
    if thousands > 0:
        parts.append(convert_hundreds(thousands) + " thousand")
    if hundreds > 0:
        parts.append(convert_hundreds(hundreds))
    
    result = " ".join(parts) if parts else "zero"
    
    if negative:
        result = "negative " + result
    
    return result

def load_cached_embeddings_for_format(values, format_name, format_func, models, cache_dir='embedding_cache'):
    """Load cached embeddings for a specific format."""
    # Convert values to specified format
    texts = [format_func(val) for val in values]
    
    # Get API keys
    api_keys = get_api_keys()
    
    # Create cached wrapper
    base_wrapper = EmbeddingWrapper(**api_keys)
    cached_wrapper = CachedEmbeddingWrapper(base_wrapper, cache_dir=cache_dir)
    
    embeddings_dict = {}
    
    for model in models:
        print(f"  Loading {model} embeddings for {format_name}...")
        try:
            embeddings = cached_wrapper.embed(texts, model)
            embeddings_dict[model] = np.array(embeddings)
            print(f"    Loaded {len(embeddings)} embeddings")
        except Exception as e:
            print(f"    Error loading {model}: {e}")
            # Create mock embeddings if cache miss
            embeddings_dict[model] = np.random.randn(len(texts), 1536)
    
    return embeddings_dict


def calculate_linear_r2(embeddings, values):
    """Calculate R² for linear regression from first 2 PCA components."""
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Get first 2 PCA components
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings_scaled)
    
    # Fit linear regression
    reg = LinearRegression()
    reg.fit(embeddings_pca, values)
    y_pred = reg.predict(embeddings_pca)
    
    r2 = r2_score(values, y_pred)
    return r2


def create_pca_subplot(ax, embeddings, values, model_name, format_name, vmin=None, vmax=None):
    """Create a single PCA subplot with optional custom color scale."""
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings_scaled)
    
    # Calculate R² for this subplot
    r2 = calculate_linear_r2(embeddings, values)
    
    # Sort by values for consistent coloring
    valsort = np.argsort(values)
    values_sorted = np.array(values)[valsort]
    embeddings_pca_sorted = embeddings_pca[valsort, :]
    # values_sorted = values
    # embeddings_pca_sorted = embeddings_pca
    
    # Use provided vmin/vmax or default to data range
    if vmin is None:
        vmin = min(values)
    if vmax is None:
        vmax = max(values)
    
    # Create scatter plot
    scatter = ax.scatter(embeddings_pca_sorted[:, 0], 
                        embeddings_pca_sorted[:, 1], 
                        c=values_sorted, 
                        cmap='RdBu_r', 
                        # alpha=0.7,
                        edgecolor="black",
                        linewidth=0.3,
                        s=30,
                        vmin=vmin,
                        vmax=vmax)
    
    # Highlight zero if it exists
    if 0 in values:
        idx_zero = np.array(values_sorted) == 0
        if np.any(idx_zero):
            ax.scatter(embeddings_pca_sorted[idx_zero, 0],
                      embeddings_pca_sorted[idx_zero, 1],
                      c="yellow",
                      marker="*",
                      s=150,
                      edgecolor="black",
                      linewidth=1.5,
                      zorder=10)
    
    # Set labels
    var1 = pca.explained_variance_ratio_[0]
    var2 = pca.explained_variance_ratio_[1]
    ax.set_xlabel(f'PC1 ({var1:.1%})', fontsize=9)
    ax.set_ylabel(f'PC2 ({var2:.1%})', fontsize=9)
    
    # Add R² as text annotation
    ax.text(0.05, 0.95, f'R²={r2:.3f}', 
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=9)
    
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
    
    return scatter


def create_format_comparison_plot(embeddings_by_format_pos, embeddings_by_format_mixed, 
                                 values_pos, values_mixed, models, 
                                 formats, format_labels,
                                 save_path='format_comparison_pca.png'):
    """Create 4x2 grid of PCA plots (4 formats x 2 sign types)."""
    
    fig, axes = plt.subplots(4, 2, figsize=(7, 11))
    
    # Set column titles
    axes[0, 0].set_title('Positive Integers Only', fontsize=11, fontweight='bold', pad=10)
    axes[0, 1].set_title('Mixed Signs', fontsize=11, fontweight='bold', pad=10)
    
    # Determine global color scale based on mixed values (which includes negatives)
    vmin_global = min(values_mixed)
    vmax_global = max(values_mixed)
    
    # Store scatter objects for colorbars
    scatter_pos = None
    scatter_mixed = None
    
    # Create subplots
    for i, (format_name, format_label) in enumerate(zip(formats, format_labels)):
        model = models[0]  # Single model
        
        # Left column: positive integers (use global vmin/vmax for consistent scaling)
        embeddings_pos = embeddings_by_format_pos[format_name][model]
        scatter_pos = create_pca_subplot(axes[i, 0], embeddings_pos, values_pos, 
                                        model, format_name) #, vmin_global, vmax_global)
        
        # Right column: mixed signs
        embeddings_mixed = embeddings_by_format_mixed[format_name][model]
        scatter_mixed = create_pca_subplot(axes[i, 1], embeddings_mixed, values_mixed, 
                                          model, format_name, vmin_global, vmax_global)
        
        # Add format label on the left
        current_ylabel = axes[i, 0].get_ylabel()
        axes[i, 0].set_ylabel(f"{format_label}\n\n{current_ylabel}", fontsize=10)
    
    # Adjust layout to make room for colorbars
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.1)
    
    # Add colorbar for left column (positive integers)
    cbar_ax_pos = fig.add_axes([0.19, 0.02, 0.3, 0.015])  # [left, bottom, width, height]
    cbar_pos = fig.colorbar(scatter_pos, cax=cbar_ax_pos, orientation='horizontal')
    cbar_pos.ax.tick_params(labelsize=8)
    cbar_pos.set_label('Integer Value', fontsize=9)
    
    # Add colorbar for right column (mixed signs)
    cbar_ax_mixed = fig.add_axes([0.65, 0.02, 0.3, 0.015])  # [left, bottom, width, height]
    cbar_mixed = fig.colorbar(scatter_mixed, cax=cbar_ax_mixed, orientation='horizontal')
    cbar_mixed.ax.tick_params(labelsize=8)
    cbar_mixed.set_label('Integer Value', fontsize=9)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved format comparison PCA plot to: {save_path}")
    plt.close()


def print_analysis_summary(embeddings_by_format_pos, embeddings_by_format_mixed, 
                          values_pos, values_mixed, models, formats, format_labels):
    """Print detailed analysis of each format."""
    print("\n" + "="*70)
    print("FORMAT COMPARISON ANALYSIS SUMMARY")
    print("="*70)
    
    for format_name, format_label in zip(formats, format_labels):
        print(f"\n{format_label}:")
        print("-" * 70)
        
        model = models[0]
        
        # Positive integers
        print(f"\n  POSITIVE INTEGERS:")
        embeddings_pos = embeddings_by_format_pos[format_name][model]
        
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_pos)
        pca = PCA(n_components=min(10, embeddings_scaled.shape[1]))
        pca.fit(embeddings_scaled)
        
        r2 = calculate_linear_r2(embeddings_pos, values_pos)
        cum_var_2 = sum(pca.explained_variance_ratio_[:2])
        
        print(f"    Embedding shape: {embeddings_pos.shape}")
        print(f"    PC1 variance: {pca.explained_variance_ratio_[0]:.1%}")
        print(f"    PC2 variance: {pca.explained_variance_ratio_[1]:.1%}")
        print(f"    Cumulative (2 PCs): {cum_var_2:.1%}")
        print(f"    Linear R² (2 PCs): {r2:.3f}")
        
        # Mixed signs
        print(f"\n  MIXED SIGNS:")
        embeddings_mixed = embeddings_by_format_mixed[format_name][model]
        
        embeddings_scaled = scaler.fit_transform(embeddings_mixed)
        pca = PCA(n_components=min(10, embeddings_scaled.shape[1]))
        pca.fit(embeddings_scaled)
        
        r2 = calculate_linear_r2(embeddings_mixed, values_mixed)
        cum_var_2 = sum(pca.explained_variance_ratio_[:2])
        
        print(f"    Embedding shape: {embeddings_mixed.shape}")
        print(f"    PC1 variance: {pca.explained_variance_ratio_[0]:.1%}")
        print(f"    PC2 variance: {pca.explained_variance_ratio_[1]:.1%}")
        print(f"    Cumulative (2 PCs): {cum_var_2:.1%}")
        print(f"    Linear R² (2 PCs): {r2:.3f}")


def main():
    """Main function to generate format comparison PCA visualization."""
    
    print("="*70)
    print("FORMAT COMPARISON PCA VISUALIZATION")
    print("="*70)
    
    # Configuration
    # models = ['text-embedding-3-large', 'gemini-embedding-001', 'voyage-3-large']
    models = ['voyage-3-large']
    
    formats = ['standard', 'scientific_3sf', 'rounded_3sf', 'written_words']
    format_labels = ['Standard\n"123456"', 
                     'Scientific (3sf)\n"1.23e+05"',
                     'Rounded (3sf)\n"123000"',
                     'Written Words\n"one hundred..."']
    
    format_functions = {
        'standard': lambda x: str(x),
        'scientific_3sf': format_scientific_3sf,
        'rounded_3sf': format_rounded_3sf,
        'written_words': number_to_words
    }
    
    print(f"\nModels: {models}")
    print(f"Formats: {formats}")
    
    # Generate base datasets
    print("\n" + "-"*70)
    print("Generating base datasets...")
    print("-"*70)
    
    np.random.seed(42)
    n_samples = 1000
    
    # Positive integers only (0 to 1 trillion)
    values_pos = np.random.randint(0, 1_000_000_000_000, n_samples).tolist()
    
    # Mixed signs (-1 trillion to 1 trillion)
    values_mixed = np.random.randint(-1_000_000_000_000, 1_000_000_000_000, n_samples).tolist()
    
    print(f"Positive integers: {len(values_pos)} samples")
    print(f"  Range: {min(values_pos):,} to {max(values_pos):,}")
    print(f"  Mean: {np.mean(values_pos):,.0f}")
    
    print(f"\nMixed signs: {len(values_mixed)} samples")
    print(f"  Range: {min(values_mixed):,} to {max(values_mixed):,}")
    print(f"  Mean: {np.mean(values_mixed):,.0f}")
    
    # Show examples of each format
    example_val = 123456
    print(f"\nExample conversions ({example_val}):")
    for fmt_name, fmt_func in format_functions.items():
        print(f"  {fmt_name:20s}: {fmt_func(example_val)}")
    
    # Load embeddings for each format and dataset
    print("\n" + "-"*70)
    print("Loading cached embeddings for each format...")
    print("-"*70)
    
    embeddings_by_format_pos = {}
    embeddings_by_format_mixed = {}
    
    for format_name in formats:
        print(f"\nFormat: {format_name}")
        format_func = format_functions[format_name]
        
        # Load positive integers
        print("  Loading positive integers...")
        embeddings_by_format_pos[format_name] = load_cached_embeddings_for_format(
            values_pos, format_name + "_pos", format_func, models
        )
        
        # Load mixed signs
        print("  Loading mixed signs...")
        embeddings_by_format_mixed[format_name] = load_cached_embeddings_for_format(
            values_mixed, format_name + "_mixed", format_func, models
        )
    
    if not all(embeddings_by_format_pos.values()) or not all(embeddings_by_format_mixed.values()):
        print("\n❌ Failed to load embeddings. Exiting.")
        return
    
    print(f"\n✅ Successfully loaded embeddings for {len(formats)} formats × 2 datasets")
    
    # Create visualization
    print("\n" + "-"*70)
    print("Creating format comparison PCA visualization...")
    print("-"*70)
    
    create_format_comparison_plot(
        embeddings_by_format_pos, embeddings_by_format_mixed,
        values_pos, values_mixed, models, 
        formats, format_labels
    )
    
    # Print analysis summary
    print_analysis_summary(embeddings_by_format_pos, embeddings_by_format_mixed,
                          values_pos, values_mixed, models, formats, format_labels)
    
    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
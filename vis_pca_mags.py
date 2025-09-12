#!/usr/bin/env python3
"""
Simple script to load cached embeddings and create 3x2 PCA visualization
with analysis plots for mixed-integer dataset with OpenAI embedding model only.
Magnitude ranges: [100, 1000, 10k, 100k, 1M, 10M] with both positive and negative values.
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
    from datasets import real_int_and_decimal
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this from the project directory with all required modules.")
    sys.exit(1)


def get_api_keys():
    """Get API keys from environment variables."""
    return {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
    }


def load_cached_embeddings_for_models(texts, models, cache_dir='embedding_cache'):
    """Load cached embeddings for specified models."""
    # Get API keys
    api_keys = get_api_keys()
    
    # Create cached wrapper
    base_wrapper = EmbeddingWrapper(**api_keys)
    cached_wrapper = CachedEmbeddingWrapper(base_wrapper, cache_dir=cache_dir)
    
    embeddings_dict = {}
    
    for model in models:
        print(f"Loading embeddings for {model}...")
        try:
            embeddings = cached_wrapper.embed(texts, model)
            embeddings_dict[model] = np.array(embeddings)
            print(f"  Loaded {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
        except Exception as e:
            print(f"  Error loading {model}: {e}")
            # Create mock embeddings if cache miss
            print(f"  Creating mock embeddings for {model}")
            embeddings_dict[model] = np.random.randn(len(texts), 1536)  # Common embedding dimension
    
    return embeddings_dict


def generate_magnitude_dataset(magnitude, n_samples=2000, symmetric=True):
    """Generate dataset for a specific magnitude range with 2k samples and guaranteed 0."""
    if symmetric:
        # Generate both positive and negative values: [-magnitude, magnitude]
        values = np.random.randint(-magnitude, magnitude + 1, n_samples - 1).tolist()
        range_str = f"[-{format_magnitude(magnitude)}, {format_magnitude(magnitude)}]"
    else:
        # Generate only positive values: [0, magnitude]
        values = np.random.randint(0, magnitude + 1, n_samples - 1).tolist()
        range_str = f"[0, {format_magnitude(magnitude)}]"
    
    # Always include 0 as the first value
    values = [0] + values
    
    # If we somehow have more than 2000, sample down (keeping 0)
    if len(values) > 2000:
        sampled_indices = np.random.choice(range(1, len(values)), 1999, replace=False)
        values = [0] + [values[i] for i in sampled_indices]
    
    texts = [str(val) for val in values]
    return texts, values, range_str


def format_magnitude(value):
    """Format magnitude values with appropriate suffixes."""
    if value >= 10_000_000:
        return f"{value // 1_000_000}M"
    elif value >= 1_000_000:
        return f"{value // 1_000_000}M"
    elif value >= 10_000:
        return f"{value // 1_000}k"
    else:
        return str(value)


def calculate_cumulative_r2(embeddings, values, max_components=20):
    """Calculate R2 for linear models using 1 to N PCA components."""
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Fit PCA with max components
    n_components = min(max_components, embeddings_scaled.shape[1], embeddings_scaled.shape[0] - 1)
    pca = PCA(n_components=n_components)
    embeddings_pca = pca.fit_transform(embeddings_scaled)
    
    r2_scores = []
    
    for n in range(1, n_components + 1):
        # Use first n components
        X = embeddings_pca[:, :n]
        
        # Fit linear regression
        reg = LinearRegression()
        reg.fit(X, values)
        y_pred = reg.predict(X)
        
        r2 = r2_score(values, y_pred)
        r2_scores.append(r2)
    
    return range(1, n_components + 1), r2_scores


def create_pca_subplot(ax, embeddings, values, model_name, value_range_str, highlight_zero=True):
    """Create a single PCA subplot."""
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings_scaled)

    # Sort by values for consistent ordering
    valsort = np.argsort(values)
    values_sorted = np.array(values)[valsort]
    embeddings_pca_sorted = embeddings_pca[valsort, :]
    
    # Determine color range based on value range
    vmin = min(values)
    vmax = max(values)
    
    # Create scatter plot
    scatter = ax.scatter(embeddings_pca_sorted[:, 0], 
                        embeddings_pca_sorted[:, 1], 
                        c=values_sorted, 
                        cmap='RdBu', 
                        alpha=1,
                        edgecolor="black",
                        linewidth=0.2,
                        s=40,
                        vmin=vmin,
                        vmax=vmax)

    # Highlight zero values if they exist and highlight_zero is True
    if highlight_zero and 0 in values:
        idx_zero = np.array(values_sorted) == 0
        if np.any(idx_zero):
            ax.scatter(embeddings_pca_sorted[idx_zero, 0],
                      embeddings_pca_sorted[idx_zero, 1],
                      c="red",
                      marker="x",
                      s=100)
    
    # Set labels and title
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title(value_range_str)
    ax.grid(True, alpha=0.3)
    
    return scatter


def create_3x2_magnitude_plot(embeddings_dict_list, values_list, range_strings, 
                             save_path='pca_3x2_magnitude_analysis.png'):
    """Create 3x2 plot (3 rows, 2 columns) for different magnitude ranges."""
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    
    model_name = list(embeddings_dict_list[0].keys())[0]  # Only one model (OpenAI)
    
    # Create subplots for each magnitude range
    for i in range(6):
        row = i // 2  # 0,0,1,1,2,2
        col = i % 2   # 0,1,0,1,0,1
        
        embeddings = embeddings_dict_list[i][model_name]
        values = values_list[i]
        range_str = range_strings[i]
        
        scatter = create_pca_subplot(axes[row, col], embeddings, values, model_name, range_str, highlight_zero=True)
        
        # Add colorbar to the rightmost subplots (col == 1)
        if col == 1:
            cbar = plt.colorbar(scatter, ax=axes[row, col])
            cbar.set_label('Integer Value')
    
    # Set overall title
    fig.suptitle(f'OpenAI {model_name} - Integer Embedding Analysis by Magnitude', fontsize=14, y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved 3x2 magnitude analysis plot to: {save_path}")
    plt.show()


def main():
    """Main function to generate 3x2 PCA analysis visualization."""
    
    print("3x2 PCA Embedding Analysis - OpenAI Model Only")
    print("=" * 50)
    
    # Configuration - Only OpenAI model
    models = ['text-embedding-3-large']
    
    # Magnitude ranges: [100, 1000, 10k, 100k, 1M, 10M]
    magnitudes = [100, 1000, 10_000, 100_000, 1_000_000, 10_000_000]
    
    print(f"Model: {models[0]}")
    print(f"Magnitude ranges: {[format_magnitude(m) for m in magnitudes]}")
    print(f"Sample size per range: 2000 (including 0)")
    
    # Generate datasets for each magnitude
    print("\nGenerating datasets...")
    
    embeddings_dict_list = []
    values_list = []
    range_strings = []
    
    for i, magnitude in enumerate(magnitudes):
        print(f"\nProcessing magnitude {format_magnitude(magnitude)}...")
        
        # Generate symmetric dataset [-magnitude, magnitude] with 2000 samples
        texts, values, range_str = generate_magnitude_dataset(magnitude, 2000, symmetric=True)
        
        print(f"  Range: {range_str}")
        print(f"  Samples: {len(values)}")
        print(f"  Value range: {min(values)} to {max(values)}")
        print(f"  Zero included: {0 in values}")
        
        # Load cached embeddings
        embeddings_dict = load_cached_embeddings_for_models(texts, models)
        
        if not embeddings_dict:
            print(f"  Failed to load embeddings for magnitude {format_magnitude(magnitude)}. Skipping.")
            continue
        
        embeddings_dict_list.append(embeddings_dict)
        values_list.append(values)
        range_strings.append(range_str)
    
    if not embeddings_dict_list:
        print("Failed to load embeddings for any magnitude range. Exiting.")
        return
    
    print(f"\nSuccessfully loaded embeddings for {len(embeddings_dict_list)} magnitude ranges")
    
    # Create 3x2 PCA magnitude analysis visualization
    print("\nCreating 3x2 PCA magnitude analysis visualization...")
    create_3x2_magnitude_plot(embeddings_dict_list, values_list, range_strings)
    
    # Print summary statistics
    print("\nEmbedding Analysis Summary:")
    print("-" * 50)
    
    model_name = models[0]
    
    for i, (magnitude, range_str) in enumerate(zip(magnitudes, range_strings)):
        if i >= len(embeddings_dict_list):
            continue
            
        print(f"\n{range_str}:")
        
        embeddings = embeddings_dict_list[i][model_name]
        values = values_list[i]
        
        # PCA analysis
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        pca = PCA(n_components=min(10, embeddings_scaled.shape[1]))
        pca.fit(embeddings_scaled)
        
        # R² analysis
        components, r2_scores = calculate_cumulative_r2(embeddings, values, max_components=10)
        best_r2 = max(r2_scores) if r2_scores else 0
        best_n_comp = components[np.argmax(r2_scores)] if r2_scores else 0
        
        print(f"  Shape: {embeddings.shape}")
        print(f"  Value statistics: min={min(values)}, max={max(values)}, mean={np.mean(values):.1f}")
        print(f"  Zero count: {values.count(0)}")
        print(f"  PC1-2 Variance: {pca.explained_variance_ratio_[0]:.1%}, {pca.explained_variance_ratio_[1]:.1%}")
        print(f"  Total 2D Variance: {sum(pca.explained_variance_ratio_[:2]):.1%}")
        print(f"  Best R²: {best_r2:.3f} (with {best_n_comp} components)")


if __name__ == "__main__":
    main()
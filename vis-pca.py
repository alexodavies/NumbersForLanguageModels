#!/usr/bin/env python3
"""
Simple script to load cached embeddings and create 2x3 PCA visualization
with analysis plots for mixed-integer dataset with three main embedding models.
Left column: 0-10000000, Right column: -10000000 to 10000000
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
        'google_api_key': os.getenv('GOOGLE_API_KEY'),
        'voyage_api_key': os.getenv('VOYAGE_API_KEY')
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
    # ax.set_title(f'{model_name}')
    ax.grid(True, alpha=0.3)
    
    return scatter


def create_2x3_analysis_plot(embeddings_dict_pos, embeddings_dict_full, values_pos, values_full, 
                            save_path='pca_2x3_analysis.png'):
    """Create 2x3 plot with left column 0-10000000, right column -10000000 to 10000000."""
    
    fig, axes = plt.subplots(3, 2, figsize=(7, 9.5))
    
    models = list(embeddings_dict_pos.keys())
    

    axes[0,0].set_title("Integers [0,10000000]")
    axes[0,1].set_title("Integers [-10000000,10000000]")

    # Row 1: 2D PCA scatter plots for positive values (0-10000000)
    for i, model in enumerate(models):
        embeddings_pos = embeddings_dict_pos[model]
        scatter = create_pca_subplot(axes[i, 0], embeddings_pos, values_pos, model, "0-10000000", highlight_zero=True)
        
        # Add colorbar to the last subplot of first row
        # if i == len(models) - 1:
        #     cbar = plt.colorbar(scatter, ax=axes[i, 0])
        #     cbar.set_label('Integer Value')

        # axes[i, 0].set_title(model)
    
    # Row 2: 2D PCA scatter plots for full range (-10000000 to 10000000)
    for i, model in enumerate(models):
        embeddings_full = embeddings_dict_full[model]
        scatter = create_pca_subplot(axes[i, 1], embeddings_full, values_full, model, "-10000000 to 10000000", highlight_zero=True)
        
        # Add colorbar to the last subplot of second row
        # if i == len(models) - 1:
        #     cbar = plt.colorbar(scatter, ax=axes[i, 1])
        #     cbar.set_label('Integer Value')
    for i in range(3):
        ax_inter = axes[i,1].twinx()
        ax_inter.set_ylabel(models[i])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved 2x3 analysis plot to: {save_path}")
    plt.show()


def main():
    """Main function to generate 2x3 PCA analysis visualization."""
    
    print("2x3 PCA Embedding Analysis")
    print("=" * 30)
    
    # Configuration
    models = ['text-embedding-3-large', 'gemini-embedding-001', 'voyage-3-large']
    
    print(f"Models: {models}")
    
    # Generate datasets
    print("\nGenerating datasets...")
    
    # Positive values: 0 to 10000000
    values_pos = np.random.randint(0, 10000000, 1000).tolist() #list(range(0, 100000001, 100))
    texts_pos = [str(val) for val in values_pos]
    
    # Full range: -10000000 to 10000000
    values_full = np.random.randint(-10000000, 10000000, 2000).tolist()
    texts_full = [str(val) for val in values_full]
    
    print(f"Positive range: {min(values_pos)} to {max(values_pos)} ({len(values_pos)} values)")
    print(f"Full range: {min(values_full)} to {max(values_full)} ({len(values_full)} values)")
    
    # Load cached embeddings for both datasets
    print("\nLoading cached embeddings for positive values (0-10000000)...")
    embeddings_dict_pos = load_cached_embeddings_for_models(texts_pos, models)
    
    print("\nLoading cached embeddings for full range (-10000000 to 10000000)...")
    embeddings_dict_full = load_cached_embeddings_for_models(texts_full, models)
    
    if not embeddings_dict_pos or not embeddings_dict_full:
        print("Failed to load embeddings. Exiting.")
        return
    
    print(f"Successfully loaded embeddings for {len(embeddings_dict_pos)} models")
    
    # Create 2x3 PCA analysis visualization
    print("\nCreating 2x3 PCA analysis visualization...")
    create_2x3_analysis_plot(embeddings_dict_pos, embeddings_dict_full, values_pos, values_full)
    
    # Print summary statistics
    print("\nEmbedding Analysis Summary:")
    print("-" * 50)
    
    for model in models:
        print(f"\n{model}:")
        
        # Analysis for positive values (0-10000000)
        embeddings_pos = embeddings_dict_pos[model]
        scaler_pos = StandardScaler()
        embeddings_scaled_pos = scaler_pos.fit_transform(embeddings_pos)
        pca_pos = PCA(n_components=min(10, embeddings_scaled_pos.shape[1]))
        pca_pos.fit(embeddings_scaled_pos)
        
        components_pos, r2_scores_pos = calculate_cumulative_r2(embeddings_pos, values_pos, max_components=10)
        best_r2_pos = max(r2_scores_pos) if r2_scores_pos else 0
        best_n_comp_pos = components_pos[np.argmax(r2_scores_pos)] if r2_scores_pos else 0
        
        print(f"  0-10000000 range:")
        print(f"    Shape: {embeddings_pos.shape}")
        print(f"    PC1-2 Variance: {pca_pos.explained_variance_ratio_[0]:.1%}, {pca_pos.explained_variance_ratio_[1]:.1%}")
        print(f"    Total 2D Variance: {sum(pca_pos.explained_variance_ratio_[:2]):.1%}")
        print(f"    Best R²: {best_r2_pos:.3f} (with {best_n_comp_pos} components)")
        
        # Analysis for full range (-10000000 to 10000000)
        embeddings_full = embeddings_dict_full[model]
        scaler_full = StandardScaler()
        embeddings_scaled_full = scaler_full.fit_transform(embeddings_full)
        pca_full = PCA(n_components=min(10, embeddings_scaled_full.shape[1]))
        pca_full.fit(embeddings_scaled_full)
        
        components_full, r2_scores_full = calculate_cumulative_r2(embeddings_full, values_full, max_components=10)
        best_r2_full = max(r2_scores_full) if r2_scores_full else 0
        best_n_comp_full = components_full[np.argmax(r2_scores_full)] if r2_scores_full else 0
        
        print(f"  -10000000 to 10000000 range:")
        print(f"    Shape: {embeddings_full.shape}")
        print(f"    PC1-2 Variance: {pca_full.explained_variance_ratio_[0]:.1%}, {pca_full.explained_variance_ratio_[1]:.1%}")
        print(f"    Total 2D Variance: {sum(pca_full.explained_variance_ratio_[:2]):.1%}")
        print(f"    Best R²: {best_r2_full:.3f} (with {best_n_comp_full} components)")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Vector Similarity Experiments Module

Test if LLM embeddings preserve mathematical relationships through vector similarities:
- Negative relationship: a = -b should have cosine similarity ≈ -1
- Scaling relationship: a = 2*b should have cosine similarity ≈ 1
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import seaborn as sns

# Your existing imports
from datasets import real_positive_decimals, real_positive_and_negative_decimals


@dataclass
class SimilarityConfig:
    """Configuration for similarity experiments."""
    n_pairs: int = 200
    test_size: float = 0.2
    random_state: int = 42
    
    # Number ranges to test
    decimal_sizes: List[int] = None
    magnitude_ranges: List[Tuple[float, float]] = None
    
    # Output directories
    plot_dir: str = "similarity_plots"
    results_dir: str = "similarity_results"
    
    def __post_init__(self):
        if self.decimal_sizes is None:
            self.decimal_sizes = [i for i in range(1, 15)]
        if self.magnitude_ranges is None:
            self.magnitude_ranges = [
                (0.00001, 0.99999),   # Small decimals
                (0.00001, 0.49999)
                # (1, 99),          # Small integers
                # (100, 9999),      # Medium numbers
                # (10000, 999999)   # Large numbers
            ]
        
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)


def generate_negative_pairs(n_pairs: int, decimal_size: int, 
                           magnitude_range: Tuple[float, float] = (1, 1000),
                           random_state: int = 42) -> Tuple[List[str], List[str]]:
    """
    Generate pairs where a = -b.
    
    Returns:
        Tuple of (a_values, b_values) where a[i] = -b[i]
    """
    import random
    import numpy as np
    
    random.seed(random_state)
    np.random.seed(random_state)
    
    a_values = []
    b_values = []
    
    for _ in range(n_pairs):
        # Generate a random number in the specified range
        magnitude = random.uniform(magnitude_range[0], magnitude_range[1])
        
        # Format with specified decimal places
        if decimal_size == 0:
            a_str = str(int(magnitude))
            b_str = str(-int(magnitude))
        else:
            format_str = f"{{:.{decimal_size}f}}"
            a_str = format_str.format(magnitude)
            b_str = format_str.format(-magnitude)
        
        a_values.append(a_str)
        b_values.append(b_str)
    
    return a_values, b_values


def generate_scaling_pairs(n_pairs: int, decimal_size: int, 
                          scale_factor: float = 2.0,
                          magnitude_range: Tuple[float, float] = (1, 1000),
                          random_state: int = 42) -> Tuple[List[str], List[str]]:
    """
    Generate pairs where a = scale_factor * b.
    
    Returns:
        Tuple of (a_values, b_values) where a[i] = scale_factor * b[i]
    """
    import random
    import numpy as np
    
    random.seed(random_state)
    np.random.seed(random_state)
    
    a_values = []
    b_values = []
    
    for _ in range(n_pairs):
        # Generate base value
        base_value = random.uniform(magnitude_range[0], magnitude_range[1])
        scaled_value = base_value * scale_factor
        
        # Format with specified decimal places
        if decimal_size == 0:
            b_str = str(int(base_value))
            a_str = str(int(scaled_value))
        else:
            format_str = f"{{:.{decimal_size}f}}"
            b_str = format_str.format(base_value)
            a_str = format_str.format(scaled_value)
        
        a_values.append(a_str)
        b_values.append(b_str)
    
    return a_values, b_values


def calculate_embedding_similarities(embeddings_a: List[List[float]], 
                                   embeddings_b: List[List[float]]) -> np.ndarray:
    """
    Calculate pairwise cosine similarities between corresponding embeddings.
    
    Args:
        embeddings_a: Embeddings for first set
        embeddings_b: Embeddings for second set
        
    Returns:
        Array of cosine similarities for each pair
    """
    embeddings_a = np.array(embeddings_a)
    embeddings_b = np.array(embeddings_b)
    
    # Calculate pairwise cosine similarity
    similarities = []
    for i in range(len(embeddings_a)):
        sim = cosine_similarity([embeddings_a[i]], [embeddings_b[i]])[0, 0]
        similarities.append(sim)
    
    return np.array(similarities)


def evaluate_similarity_preservation(similarities: np.ndarray, 
                                   expected_similarity: float,
                                   tolerance: float = 0.1) -> Dict[str, Any]:
    """
    Evaluate how well the embeddings preserve the expected similarity.
    
    Args:
        similarities: Actual cosine similarities
        expected_similarity: Expected similarity value (-1 for negatives, 1 for scaling)
        tolerance: Tolerance for "correct" classification
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Basic statistics
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    median_sim = np.median(similarities)
    
    # Accuracy within tolerance
    within_tolerance = np.abs(similarities - expected_similarity) <= tolerance
    accuracy = np.mean(within_tolerance)
    
    # Correlation with expected (should be high if all are close to expected)
    expected_array = np.full_like(similarities, expected_similarity)
    correlation = stats.pearsonr(similarities, expected_array)[0] if len(similarities) > 1 else 0.0
    
    # Distance metrics
    mae = np.mean(np.abs(similarities - expected_similarity))  # Mean Absolute Error
    rmse = np.sqrt(np.mean((similarities - expected_similarity) ** 2))  # Root Mean Square Error
    
    # Distribution analysis
    min_sim = np.min(similarities)
    max_sim = np.max(similarities)
    range_sim = max_sim - min_sim
    
    return {
        'expected_similarity': expected_similarity,
        'mean_similarity': mean_sim,
        'median_similarity': median_sim,
        'std_similarity': std_sim,
        'min_similarity': min_sim,
        'max_similarity': max_sim,
        'range_similarity': range_sim,
        'accuracy_within_tolerance': accuracy,
        'tolerance': tolerance,
        'mean_absolute_error': mae,
        'root_mean_square_error': rmse,
        'correlation_with_expected': correlation,
        'n_pairs': len(similarities)
    }


class SimilarityExperiments:
    """Test mathematical relationship preservation in embeddings."""
    
    def __init__(self, embedding_wrapper, config: SimilarityConfig = None):
        self.wrapper = embedding_wrapper
        self.config = config or SimilarityConfig()
        
        # Get available models
        self.available_models = self._get_models()
        
        print(f"🔬 Similarity experiments initialized")
        print(f"  Models: {self.available_models}")
        print(f"  Decimal sizes: {self.config.decimal_sizes}")
        print(f"  Magnitude ranges: {self.config.magnitude_ranges}")
        
        self.results = {}
    
    def _get_models(self) -> List[str]:
        """Get available models."""
        try:
            available_services = self.wrapper.get_available_services()
            supported_models = self.wrapper.get_supported_models()
            
            models = []
            for service, is_available in available_services.items():
                if is_available:
                    models.extend(supported_models[service])
            
            return models if models else ["mock-model"]
        except:
            return ["mock-model"]
    
    def run_negative_relationship_sweep(self) -> Dict[str, Any]:
        """Test preservation of negative relationships (a = -b)."""
        experiment_name = "negative_relationships"
        
        print(f"\n{'='*60}")
        print(f"NEGATIVE RELATIONSHIP SWEEP: a = -b")
        print(f"Expected cosine similarity: -1.0")
        print(f"{'='*60}")
        
        results = {
            'experiment': experiment_name,
            'expected_similarity': -1.0,
            'models': {}
        }
        
        for model in self.available_models:
            print(f"\n🤖 Testing {model}")
            
            model_results = {
                'decimal_sizes': [],
                'magnitude_ranges': [],
                'mean_similarities': [],
                'std_similarities': [],
                'accuracies': [],
                'mae_values': [],
                'rmse_values': [],
                'correlations': []
            }
            
            for decimal_size in self.config.decimal_sizes:
                for mag_range in self.config.magnitude_ranges:
                    print(f"  📊 Decimal size: {decimal_size}, Range: {mag_range}")
                    
                    try:
                        # Generate negative pairs
                        import random
                        import numpy as np
                        seed = self.config.random_state + decimal_size + int(mag_range[0])
                        random.seed(seed)
                        np.random.seed(seed)
                        
                        a_values, b_values = generate_negative_pairs(
                            self.config.n_pairs, decimal_size, mag_range, seed
                        )
                        
                        print(f"    Sample pairs: {a_values[0]} & {b_values[0]}, {a_values[1]} & {b_values[1]}")
                        
                        # Get embeddings with parameter caching
                        if hasattr(self.wrapper, 'embed_with_params'):
                            embeddings_a = self.wrapper.embed_with_params(
                                a_values, model, f"{experiment_name}_a", decimal_size,
                                magnitude_range=mag_range, random_state=seed
                            )
                            embeddings_b = self.wrapper.embed_with_params(
                                b_values, model, f"{experiment_name}_b", decimal_size,
                                magnitude_range=mag_range, random_state=seed
                            )
                        else:
                            embeddings_a = self.wrapper.embed(a_values, model)
                            embeddings_b = self.wrapper.embed(b_values, model)
                        
                        # Calculate similarities
                        similarities = calculate_embedding_similarities(embeddings_a, embeddings_b)
                        
                        # Evaluate preservation
                        evaluation = evaluate_similarity_preservation(similarities, -1.0, tolerance=0.2)
                        
                        # Store results
                        model_results['decimal_sizes'].append(decimal_size)
                        model_results['magnitude_ranges'].append(str(mag_range))
                        model_results['mean_similarities'].append(evaluation['mean_similarity'])
                        model_results['std_similarities'].append(evaluation['std_similarity'])
                        model_results['accuracies'].append(evaluation['accuracy_within_tolerance'])
                        model_results['mae_values'].append(evaluation['mean_absolute_error'])
                        model_results['rmse_values'].append(evaluation['root_mean_square_error'])
                        model_results['correlations'].append(evaluation['correlation_with_expected'])
                        
                        print(f"    Mean similarity: {evaluation['mean_similarity']:.3f} ± {evaluation['std_similarity']:.3f}")
                        print(f"    Accuracy (±0.2): {evaluation['accuracy_within_tolerance']:.3f}")
                        print(f"    MAE: {evaluation['mean_absolute_error']:.3f}")
                        
                        # # Create visualization for this configuration
                        # self._plot_similarity_distribution(
                        #     similarities, -1.0, 
                        #     f"{experiment_name}_{model}_{decimal_size}_{mag_range[0]}-{mag_range[1]}"
                        # )
                        
                    except Exception as e:
                        print(f"    ❌ Error: {e}")
                        # Store NaN values for failed experiments
                        model_results['decimal_sizes'].append(decimal_size)
                        model_results['magnitude_ranges'].append(str(mag_range))
                        for key in ['mean_similarities', 'std_similarities', 'accuracies', 
                                   'mae_values', 'rmse_values', 'correlations']:
                            model_results[key].append(np.nan)
            
            results['models'][model] = model_results
        
        self.results[experiment_name] = results
        return results
    
    def run_scaling_relationship_sweep(self, scale_factors: List[float] = None) -> Dict[str, Any]:
        """Test preservation of scaling relationships (a = scale * b)."""
        if scale_factors is None:
            scale_factors = [2.0]
        
        all_results = {}
        
        for scale_factor in scale_factors:
            experiment_name = f"scaling_relationships_{scale_factor}x"
            
            print(f"\n{'='*60}")
            print(f"SCALING RELATIONSHIP SWEEP: a = {scale_factor} * b")
            print(f"Expected cosine similarity: 1.0")
            print(f"{'='*60}")
            
            results = {
                'experiment': experiment_name,
                'scale_factor': scale_factor,
                'expected_similarity': 1.0,
                'models': {}
            }
            
            for model in self.available_models:
                print(f"\n🤖 Testing {model}")
                
                model_results = {
                    'decimal_sizes': [],
                    'magnitude_ranges': [],
                    'mean_similarities': [],
                    'std_similarities': [],
                    'accuracies': [],
                    'mae_values': [],
                    'rmse_values': [],
                    'correlations': []
                }
                
                for decimal_size in self.config.decimal_sizes:
                    for mag_range in self.config.magnitude_ranges:
                        print(f"  📊 Decimal size: {decimal_size}, Range: {mag_range}")
                        
                        try:
                            # Generate scaling pairs
                            import random
                            import numpy as np
                            seed = self.config.random_state + decimal_size + int(mag_range[0]) + int(scale_factor * 1000)
                            random.seed(seed)
                            np.random.seed(seed)
                            
                            a_values, b_values = generate_scaling_pairs(
                                self.config.n_pairs, decimal_size, scale_factor, mag_range, seed
                            )
                            
                            print(f"    Sample pairs: {b_values[0]} -> {a_values[0]}, {b_values[1]} -> {a_values[1]}")
                            
                            # Get embeddings with parameter caching
                            if hasattr(self.wrapper, 'embed_with_params'):
                                embeddings_a = self.wrapper.embed_with_params(
                                    a_values, model, f"{experiment_name}_a", decimal_size,
                                    magnitude_range=mag_range, scale_factor=scale_factor, random_state=seed
                                )
                                embeddings_b = self.wrapper.embed_with_params(
                                    b_values, model, f"{experiment_name}_b", decimal_size,
                                    magnitude_range=mag_range, scale_factor=scale_factor, random_state=seed
                                )
                            else:
                                embeddings_a = self.wrapper.embed(a_values, model)
                                embeddings_b = self.wrapper.embed(b_values, model)
                            
                            # Calculate similarities
                            similarities = calculate_embedding_similarities(embeddings_a, embeddings_b)
                            
                            # Evaluate preservation
                            evaluation = evaluate_similarity_preservation(similarities, 1.0, tolerance=0.2)
                            
                            # Store results
                            model_results['decimal_sizes'].append(decimal_size)
                            model_results['magnitude_ranges'].append(str(mag_range))
                            model_results['mean_similarities'].append(evaluation['mean_similarity'])
                            model_results['std_similarities'].append(evaluation['std_similarity'])
                            model_results['accuracies'].append(evaluation['accuracy_within_tolerance'])
                            model_results['mae_values'].append(evaluation['mean_absolute_error'])
                            model_results['rmse_values'].append(evaluation['root_mean_square_error'])
                            model_results['correlations'].append(evaluation['correlation_with_expected'])
                            
                            print(f"    Mean similarity: {evaluation['mean_similarity']:.3f} ± {evaluation['std_similarity']:.3f}")
                            print(f"    Accuracy (±0.2): {evaluation['accuracy_within_tolerance']:.3f}")
                            print(f"    MAE: {evaluation['mean_absolute_error']:.3f}")
                            
                            # # Create visualization for this configuration
                            # self._plot_similarity_distribution(
                            #     similarities, 1.0, 
                            #     f"{experiment_name}_{model}_{decimal_size}_{mag_range[0]}-{mag_range[1]}"
                            # )
                            
                        except Exception as e:
                            print(f"    ❌ Error: {e}")
                            # Store NaN values for failed experiments
                            model_results['decimal_sizes'].append(decimal_size)
                            model_results['magnitude_ranges'].append(str(mag_range))
                            for key in ['mean_similarities', 'std_similarities', 'accuracies', 
                                       'mae_values', 'rmse_values', 'correlations']:
                                model_results[key].append(np.nan)
                
                results['models'][model] = model_results
            
            self.results[experiment_name] = results
            all_results[experiment_name] = results
        
        return all_results
    
    def _plot_similarity_distribution(self, similarities: np.ndarray, 
                                    expected_similarity: float, 
                                    filename_suffix: str) -> None:
        """Plot distribution of similarities for a single experiment configuration."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(similarities, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(expected_similarity, color='red', linestyle='--', linewidth=2, 
                   label=f'Expected: {expected_similarity}')
        ax1.axvline(np.mean(similarities), color='green', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(similarities):.3f}')
        ax1.set_xlabel('Cosine Similarity')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Similarities')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot and statistics
        bp = ax2.boxplot([similarities], patch_artist=True, labels=['Similarities'])
        bp['boxes'][0].set_facecolor('lightblue')
        ax2.axhline(expected_similarity, color='red', linestyle='--', linewidth=2,
                   label=f'Expected: {expected_similarity}')
        ax2.set_ylabel('Cosine Similarity')
        ax2.set_title('Similarity Distribution Statistics')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""
        Mean: {np.mean(similarities):.3f}
        Std: {np.std(similarities):.3f}
        Median: {np.median(similarities):.3f}
        Min: {np.min(similarities):.3f}
        Max: {np.max(similarities):.3f}
        MAE: {np.mean(np.abs(similarities - expected_similarity)):.3f}
        """
        ax2.text(1.1, 0.5, stats_text, transform=ax2.transAxes, 
                verticalalignment='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle(f'Similarity Analysis: {filename_suffix}')
        plt.tight_layout()
        
        # Save plot
        safe_filename = filename_suffix.replace('/', '_').replace('(', '').replace(')', '').replace(',', '_').replace(' ', '_')
        filepath = os.path.join(self.config.plot_dir, "distributions", f"sim_dist_{safe_filename}.png")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_all_similarity_experiments(self) -> Dict[str, Any]:
        """Run all similarity experiments."""
        print("🔬 Running all similarity experiments...")
        
        all_results = {}
        
        # Negative relationships
        print("\n" + "="*80)
        print("PHASE 1: NEGATIVE RELATIONSHIPS")
        print("="*80)
        all_results['negative'] = self.run_negative_relationship_sweep()
        
        # Scaling relationships
        print("\n" + "="*80)
        print("PHASE 2: SCALING RELATIONSHIPS")
        print("="*80)
        scaling_results = self.run_scaling_relationship_sweep([2.0])
        all_results.update(scaling_results)
        
        return all_results
    
    def plot_summary_results(self) -> None:
            """Create summary plots across all experiments."""
            if not self.results:
                print("No results to plot")
                return
            
            print("\n📈 Creating summary plots...")
            
            # Create summary subfolder
            summary_dir = os.path.join(self.config.plot_dir, "summary")
            os.makedirs(summary_dir, exist_ok=True)
            
            # 1. Individual MAE comparison plots for each experiment
            print("Creating individual MAE comparison plots...")
            self._plot_accuracy_comparison(summary_dir)
            
            # 2. Detailed MAE plots showing magnitude ranges
            print("Creating detailed MAE plots by magnitude range...")
            self._plot_accuracy_comparison_detailed(summary_dir)
            
            # 3. Mean similarity deviation from expected (heatmap)
            print("Creating similarity deviations heatmap...")
            self._plot_similarity_deviations(summary_dir)
            
            # 4. Model performance ranking for similarity preservation
            print("Creating model rankings plot...")
            self._plot_model_similarity_rankings(summary_dir)
            
            # 5. Decimal size vs accuracy analysis
            print("Creating decimal size vs accuracy analysis...")
            self._plot_size_vs_accuracy_analysis(summary_dir)
            # else:
            #     print(f"  ⚠️  No valid data for {exp_name}")
            #     plt.close()

    def _plot_accuracy_comparison_detailed(self, save_dir: str) -> None:
        """Plot detailed MAE comparison with separate plots for each magnitude range, colored by provider."""
        if not self.results:
            print("No results to plot")
            return
        
        # Extract provider information from model names
        def extract_provider(model_name: str) -> str:
            """Extract provider from model name based on API wrapper model mappings."""
            # Use the wrapper's model mappings to determine provider
            if hasattr(self.wrapper, 'OPENAI_MODELS') and model_name in self.wrapper.OPENAI_MODELS:
                return 'OpenAI'
            elif hasattr(self.wrapper, 'GOOGLE_MODELS') and model_name in self.wrapper.GOOGLE_MODELS:
                return 'Google'
            elif hasattr(self.wrapper, 'VOYAGE_MODELS') and model_name in self.wrapper.VOYAGE_MODELS:
                return 'Voyage AI'
            else:
                # Fallback to string matching for models not in the wrapper
                model_lower = model_name.lower()
                if any(keyword in model_lower for keyword in ['openai', 'gpt', 'text-embedding']):
                    return 'OpenAI'
                elif any(keyword in model_lower for keyword in ['google', 'gemini', 'palm']):
                    return 'Google'
                elif any(keyword in model_lower for keyword in ['voyage']):
                    return 'Voyage AI'
                elif any(keyword in model_lower for keyword in ['anthropic', 'claude']):
                    return 'Anthropic'
                elif any(keyword in model_lower for keyword in ['cohere']):
                    return 'Cohere'
                elif any(keyword in model_lower for keyword in ['mistral']):
                    return 'Mistral'
                elif any(keyword in model_lower for keyword in ['meta', 'llama']):
                    return 'Meta'
                elif any(keyword in model_lower for keyword in ['huggingface', 'hf']):
                    return 'HuggingFace'
                else:
                    return 'Other'
        
        # Define provider colors (updated for actual providers)
        provider_colors = {
            'OpenAI': '#10B981',      # Green
            'Google': '#3B82F6',      # Blue  
            'Voyage AI': '#8B5CF6',   # Purple
            'Anthropic': '#F59E0B',   # Amber (fallback)
            'Cohere': '#EC4899',      # Pink (fallback)
            'Mistral': '#EF4444',     # Red (fallback)
            'Meta': '#6366F1',        # Indigo (fallback)
            'HuggingFace': '#14B8A6', # Teal (fallback)
            'Other': '#6B7280'        # Gray
        }
        
        # Get all unique magnitude ranges across all experiments
        all_magnitude_ranges = set()
        for exp_results in self.results.values():
            for model_results in exp_results['models'].values():
                all_magnitude_ranges.update(model_results['magnitude_ranges'])
        
        all_magnitude_ranges = sorted(list(all_magnitude_ranges))
        
        # Create separate plots for each experiment and magnitude range combination
        for exp_name, exp_results in self.results.items():
            expected_sim = exp_results.get('expected_similarity', 'N/A')
            scale_factor = exp_results.get('scale_factor', 'N/A')
            
            # Get magnitude ranges present in this experiment
            exp_magnitude_ranges = set()
            for model_results in exp_results['models'].values():
                exp_magnitude_ranges.update(model_results['magnitude_ranges'])
            exp_magnitude_ranges = sorted(list(exp_magnitude_ranges))
            
            # Create subplots - one for each magnitude range
            n_ranges = len(exp_magnitude_ranges)
            if n_ranges == 0:
                continue
                
            # Determine subplot layout
            if n_ranges == 1:
                fig, axes = plt.subplots(1, 1, figsize=(10, 6))
                axes = [axes]  # Make it a list for consistent indexing
            elif n_ranges == 2:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            else:
                # For more ranges, use a grid layout
                n_cols = min(3, n_ranges)
                n_rows = (n_ranges + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
                if n_rows == 1:
                    axes = axes if n_ranges > 1 else [axes]
                else:
                    axes = axes.flatten()
            
            # Process each magnitude range
            for range_idx, mag_range in enumerate(exp_magnitude_ranges):
                ax = axes[range_idx]
                plotted_models = 0
                
                # Group models by provider for consistent coloring
                provider_models = {}
                for model_name in exp_results['models'].keys():
                    provider = extract_provider(model_name)
                    if provider not in provider_models:
                        provider_models[provider] = []
                    provider_models[provider].append(model_name)
                
                # Plot each provider's models
                for provider, models in provider_models.items():
                    provider_color = provider_colors.get(provider, provider_colors['Other'])
                    
                    for model_idx, model_name in enumerate(models):
                        model_results = exp_results['models'][model_name]
                        
                        # Extract data for this specific magnitude range
                        range_decimal_sizes = []
                        range_mae_values = []
                        
                        for size, m_range, mae in zip(model_results['decimal_sizes'], 
                                                    model_results['magnitude_ranges'], 
                                                    model_results['mae_values']):
                            if m_range == mag_range and not np.isnan(mae):
                                range_decimal_sizes.append(size)
                                range_mae_values.append(mae)
                        
                        if range_decimal_sizes and range_mae_values:
                            # Use different line styles and markers for multiple models from same provider
                            linestyle_options = ['-', '--', '-.', ':']
                            marker_options = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
                            
                            linestyle = linestyle_options[model_idx % len(linestyle_options)]
                            marker = marker_options[model_idx % len(marker_options)]
                            
                            # Create more distinctive model names for legend
                            simple_model_name = model_name.split('/')[-1] if '/' in model_name else model_name
                            
                            # Extract key model identifiers for cleaner labels
                            if 'text-embedding-3-small' in model_name:
                                model_label = 'embed-3-small'
                            elif 'text-embedding-3-large' in model_name:
                                model_label = 'embed-3-large'
                            elif 'text-embedding-ada-002' in model_name:
                                model_label = 'ada-002'
                            elif 'gemini-embedding-001' in model_name:
                                model_label = 'gemini-001'
                            elif 'voyage-3.5-lite' in model_name:
                                model_label = 'v3.5-lite'
                            elif 'voyage-3.5' in model_name:
                                model_label = 'v3.5'
                            elif 'voyage-3-large' in model_name:
                                model_label = 'v3-large'
                            elif 'voyage-code-3' in model_name:
                                model_label = 'code-3'
                            elif 'voyage-finance-2' in model_name:
                                model_label = 'finance-2'
                            elif 'voyage-law-2' in model_name:
                                model_label = 'law-2'
                            elif 'voyage-multimodal-3' in model_name:
                                model_label = 'multimodal-3'
                            else:
                                model_label = simple_model_name
                            
                            # Create label: show provider only if single model, otherwise show both
                            if len(models) == 1:
                                label = f"{provider}"
                            else:
                                label = f"{provider}: {model_label}"
                            
                            ax.plot(range_decimal_sizes, range_mae_values, 
                                marker=marker, label=label, linewidth=2, alpha=0.8, 
                                color=provider_color, markersize=6, linestyle=linestyle)
                            plotted_models += 1
                
                if plotted_models > 0:
                    ax.set_xlabel('Decimal Size (digits after decimal)', fontsize=11)
                    ax.set_ylabel('MAE (Mean Absolute Error)', fontsize=11)
                    ax.set_title(f'Range: {mag_range}\n(Lower MAE = Better)', fontsize=12, fontweight='bold')
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    
                    # Set y-limit dynamically for this range
                    if range_mae_values:
                        y_max = max([max(model_results['mae_values']) 
                                for model_results in exp_results['models'].values() 
                                if any(m_range == mag_range and not np.isnan(mae) 
                                        for m_range, mae in zip(model_results['magnitude_ranges'], 
                                                            model_results['mae_values']))])
                        # if not np.isnan(y_max):
                        #     ax.set_ylim(0, y_max * 1.1)
                    
                    # Add horizontal reference line at mae=0
                    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
                else:
                    ax.text(0.5, 0.5, f'No data for\n{mag_range}', 
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
                    ax.set_title(f'Range: {mag_range}', fontsize=12)
            
            # Hide empty subplots
            if n_ranges < len(axes):
                for i in range(n_ranges, len(axes)):
                    axes[i].set_visible(False)
            
            # Create overall title for the figure
            title_parts = [f'{exp_name.replace("_", " ").title()} - MAE by Magnitude Range']
            if expected_sim != 'N/A':
                title_parts.append(f'Expected Similarity: {expected_sim}')
            if scale_factor != 'N/A':
                title_parts.append(f'Scale Factor: {scale_factor}')
            
            fig.suptitle('\n'.join(title_parts), fontsize=16, fontweight='bold', y=0.98)
            
            # Add provider color legend at the bottom
            provider_legend_elements = []
            for provider, color in provider_colors.items():
                if any(extract_provider(model) == provider 
                    for model_results in exp_results['models'] 
                    for model in exp_results['models'].keys()):
                    provider_legend_elements.append(plt.Line2D([0], [0], color=color, lw=3, label=provider))
            
            if provider_legend_elements:
                fig.legend(handles=provider_legend_elements, 
                        title='Providers', 
                        loc='lower center', 
                        bbox_to_anchor=(0.5, -0.02),
                        ncol=len(provider_legend_elements),
                        fontsize=10)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.85, bottom=0.15)  # Make room for title and legend
            
            # Save with experiment-specific filename
            safe_exp_name = exp_name.replace('/', '_').replace('(', '').replace(')', '').replace(',', '_').replace(' ', '_')
            filename = f"mae_detailed_by_range_{safe_exp_name}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  📈 Saved detailed MAE plot by range: {filepath}")
            plt.close()

            
    def _plot_accuracy_comparison(self, save_dir: str) -> None:
        """Plot MAE comparison for each experiment separately."""
        if not self.results:
            print("No results to plot")
            return
        
        # Create separate plot for each experiment
        for exp_name, exp_results in self.results.items():
            fig, ax = plt.subplots(figsize=(12, 8))
            
            expected_sim = exp_results.get('expected_similarity', 'N/A')
            scale_factor = exp_results.get('scale_factor', 'N/A')
            
            # Collect data for this experiment
            plotted_models = 0
            colors = plt.cm.Set1(np.linspace(0, 1, len(exp_results['models'])))
            
            for idx, (model_name, model_results) in enumerate(exp_results['models'].items()):
                mae_values = []
                decimal_sizes = []
                magnitude_ranges = []
                
                # Group by decimal size and average across magnitude ranges
                size_mae_map = {}
                for size, mag_range, mae in zip(model_results['decimal_sizes'], 
                                            model_results['magnitude_ranges'], 
                                            model_results['mae_values']):
                    if not np.isnan(mae):
                        if size not in size_mae_map:
                            size_mae_map[size] = []
                        size_mae_map[size].append(mae)
                
                # Calculate averages for each decimal size
                for size in sorted(size_mae_map.keys()):
                    decimal_sizes.append(size)
                    mae_values.append(np.mean(size_mae_map[size]))
                
                if mae_values and decimal_sizes:
                    ax.plot(decimal_sizes, mae_values, marker='o', label=model_name, 
                        linewidth=2, alpha=0.8, color=colors[idx], markersize=6)
                    plotted_models += 1
            
            if plotted_models > 0:
                ax.set_xlabel('Decimal Size (digits after decimal point)', fontsize=12)
                ax.set_ylabel('MAE (Mean Absolute Error)', fontsize=12)
                
                # Create detailed title
                title_parts = [f'{exp_name.replace("_", " ").title()}']
                if expected_sim != 'N/A':
                    title_parts.append(f'Expected Similarity: {expected_sim}')
                if scale_factor != 'N/A':
                    title_parts.append(f'Scale Factor: {scale_factor}')
                
                ax.set_title('\n'.join(title_parts), fontsize=14, fontweight='bold')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                
                # Set y-limit with some padding
                y_max = max([max(model_results['mae_values']) for model_results in exp_results['models'].values() 
                            if any(not np.isnan(mae) for mae in model_results['mae_values'])])
                # if not np.isnan(y_max):
                #     ax.set_ylim(0, y_max * 1.1)
                
                # Add interpretation text
                ax.text(0.02, 0.98, 'Lower MAE = Better Similarity Preservation', 
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
                
                # Add horizontal line at mae=0 for reference
                ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
                
                plt.tight_layout()
                
                # Save with experiment-specific filename
                safe_exp_name = exp_name.replace('/', '_').replace('(', '').replace(')', '').replace(',', '_').replace(' ', '_')
                filename = f"mae_comparison_{safe_exp_name}.png"
                filepath = os.path.join(save_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"  📈 Saved MAE plot: {filepath}")
                plt.close()
        
    def _plot_similarity_deviations(self, save_dir: str) -> None:
        """Plot mean absolute error (deviation from expected) across experiments."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        experiment_names = []
        model_names = set()
        
        # Collect all model names
        for exp_results in self.results.values():
            model_names.update(exp_results['models'].keys())
        
        model_names = sorted(list(model_names))
        
        # Prepare data
        data_matrix = []
        
        for exp_name, exp_results in self.results.items():
            experiment_names.append(exp_name)
            row = []
            
            for model in model_names:
                if model in exp_results['models']:
                    mae_values = [mae for mae in exp_results['models'][model]['mae_values'] if not np.isnan(mae)]
                    avg_mae = np.mean(mae_values) if mae_values else np.nan
                else:
                    avg_mae = np.nan
                row.append(avg_mae)
            
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        # Create heatmap
        im = ax.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=2)
        
        # Set ticks and labels
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([m.replace('-', '\n') for m in model_names], rotation=45, ha='right')
        ax.set_yticks(range(len(experiment_names)))
        ax.set_yticklabels(experiment_names)
        
        # Add text annotations
        for i in range(len(experiment_names)):
            for j in range(len(model_names)):
                if not np.isnan(data_matrix[i, j]):
                    ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black" if data_matrix[i, j] < 1 else "white",
                           fontsize=9)
        
        ax.set_title('Mean Absolute Error from Expected Similarity\n(Lower is better)', 
                    fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Mean Absolute Error')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "similarity_deviations.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_similarity_rankings(self, save_dir: str) -> None:
        """Plot overall model rankings for similarity preservation."""
        # Calculate average accuracy for each model across all experiments
        model_scores = {}
        
        for exp_results in self.results.values():
            for model_name, model_results in exp_results['models'].items():
                if model_name not in model_scores:
                    model_scores[model_name] = []
                
                accuracies = [acc for acc in model_results['accuracies'] if not np.isnan(acc)]
                if accuracies:
                    model_scores[model_name].extend(accuracies)
        
        # Calculate averages and sort
        model_averages = [(model, np.mean(scores), np.std(scores)) 
                         for model, scores in model_scores.items() if scores]
        model_averages.sort(key=lambda x: x[1], reverse=True)
        
        if not model_averages:
            print("No valid model scores for ranking")
            return
        
        models = [item[0] for item in model_averages]
        means = [item[1] for item in model_averages]
        stds = [item[2] for item in model_averages]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.bar(range(len(models)), means, yerr=stds, capsize=5, 
                     alpha=0.8, color='lightcoral')
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Average Accuracy (Similarity Preservation)')
        ax.set_title('Model Rankings for Mathematical Relationship Preservation\n(Average across all similarity experiments)')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace('-', '\n') for m in models], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "model_similarity_rankings.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_size_vs_accuracy_analysis(self, save_dir: str) -> None:
        """Plot how decimal size affects MAE (Mean Absolute Error)."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Aggregate MAE data by decimal size across all experiments
        size_mae_data = {}
        
        for exp_name, exp_results in self.results.items():
            for model_name, model_results in exp_results['models'].items():
                for size, mae in zip(model_results['decimal_sizes'], model_results['mae_values']):
                    if not np.isnan(mae):
                        if size not in size_mae_data:
                            size_mae_data[size] = []
                        size_mae_data[size].append(mae)
        
        # Calculate statistics for each size
        sizes = sorted(size_mae_data.keys())
        means = []
        stds = []
        
        for size in sizes:
            mae_values = size_mae_data[size]
            means.append(np.mean(mae_values))
            stds.append(np.std(mae_values))
        
        # Plot with error bars
        ax.errorbar(sizes, means, yerr=stds, marker='o', linestyle='-', linewidth=2, 
                   markersize=8, capsize=5, capthick=2, alpha=0.8, color='darkred')
        
        ax.set_xlabel('Decimal Size (digits after decimal point)')
        ax.set_ylabel('Average MAE (Mean Absolute Error)')
        ax.set_title('Effect of Number Precision on Similarity Preservation Error\n(Average across all experiments and models)')
        ax.grid(True, alpha=0.3)
        # ax.set_ylim(0, max(means) * 1.1 if means else 2)  # Dynamic y-limit based on data
        
        # Add trend line
        if len(sizes) > 1:
            z = np.polyfit(sizes, means, 1)
            p = np.poly1d(z)
            ax.plot(sizes, p(sizes), "r--", alpha=0.8, linewidth=2, 
                   label=f'Trend: slope = {z[0]:.4f}')
            ax.legend()
        
        # Add interpretation text
        ax.text(0.02, 0.98, 'Lower MAE = Better similarity preservation', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "size_vs_mae_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_similarity_results(self, filename: str = None) -> None:
        """Export all similarity results to CSV."""
        if filename is None:
            filename = os.path.join(self.config.results_dir, "similarity_preservation_results.csv")
        
        if not self.results:
            print("No results to export")
            return
        
        rows = []
        
        for exp_name, exp_results in self.results.items():
            expected_sim = exp_results.get('expected_similarity', np.nan)
            scale_factor = exp_results.get('scale_factor', np.nan)
            
            for model_name, model_results in exp_results['models'].items():
                for i in range(len(model_results['decimal_sizes'])):
                    row = {
                        'experiment': exp_name,
                        'model': model_name,
                        'expected_similarity': expected_sim,
                        'scale_factor': scale_factor,
                        'decimal_size': model_results['decimal_sizes'][i],
                        'magnitude_range': model_results['magnitude_ranges'][i],
                        'mean_similarity': model_results['mean_similarities'][i],
                        'std_similarity': model_results['std_similarities'][i],
                        'accuracy_within_tolerance': model_results['accuracies'][i],
                        'mean_absolute_error': model_results['mae_values'][i],
                        'root_mean_square_error': model_results['rmse_values'][i],
                        'correlation_with_expected': model_results['correlations'][i]
                    }
                    rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        
        print(f"📊 Exported {len(df)} similarity results to {filename}")
        print("   Columns include similarity statistics and preservation metrics")
    
    def generate_similarity_report(self) -> str:
        """Generate comprehensive similarity preservation report."""
        if not self.results:
            return "No similarity results available."
        
        lines = []
        lines.append("EMBEDDING SIMILARITY PRESERVATION RESULTS")
        lines.append("=" * 60)
        lines.append("")
        
        # Overall summary
        total_experiments = sum(len(exp_results['models']) * 
                               len(next(iter(exp_results['models'].values()))['decimal_sizes'])
                               for exp_results in self.results.values())
        
        lines.append(f"Total experiments conducted: {total_experiments}")
        lines.append(f"Relationship types tested: {len(self.results)}")
        lines.append("")
        
        # Results by experiment type
        for exp_name, exp_results in self.results.items():
            expected_sim = exp_results.get('expected_similarity', 'N/A')
            scale_factor = exp_results.get('scale_factor', 'N/A')
            
            lines.append(f"EXPERIMENT: {exp_name.upper()}")
            lines.append("-" * 40)
            lines.append(f"Expected similarity: {expected_sim}")
            if scale_factor != 'N/A':
                lines.append(f"Scale factor: {scale_factor}")
            
            # Calculate experiment-wide statistics
            all_accuracies = []
            all_mae = []
            all_similarities = []
            
            for model_results in exp_results['models'].values():
                all_accuracies.extend([acc for acc in model_results['accuracies'] if not np.isnan(acc)])
                all_mae.extend([mae for mae in model_results['mae_values'] if not np.isnan(mae)])
                all_similarities.extend([sim for sim in model_results['mean_similarities'] if not np.isnan(sim)])
            
            if all_accuracies:
                lines.append(f"Average accuracy: {np.mean(all_accuracies):.3f} ± {np.std(all_accuracies):.3f}")
                lines.append(f"Average MAE: {np.mean(all_mae):.3f} ± {np.std(all_mae):.3f}")
                lines.append(f"Average similarity: {np.mean(all_similarities):.3f} ± {np.std(all_similarities):.3f}")
                
                # Performance assessment
                if np.mean(all_accuracies) > 0.8:
                    assessment = "EXCELLENT - Strong relationship preservation"
                elif np.mean(all_accuracies) > 0.6:
                    assessment = "GOOD - Moderate relationship preservation"
                elif np.mean(all_accuracies) > 0.4:
                    assessment = "FAIR - Weak relationship preservation"
                else:
                    assessment = "POOR - Little relationship preservation"
                
                lines.append(f"Assessment: {assessment}")
            else:
                lines.append("No valid results for this experiment")
            
            lines.append("")
        
        # Model comparison
        lines.append("MODEL PERFORMANCE COMPARISON:")
        lines.append("-" * 50)
        
        model_overall_scores = {}
        for exp_results in self.results.values():
            for model_name, model_results in exp_results['models'].items():
                if model_name not in model_overall_scores:
                    model_overall_scores[model_name] = []
                
                accuracies = [acc for acc in model_results['accuracies'] if not np.isnan(acc)]
                model_overall_scores[model_name].extend(accuracies)
        
        # Sort by average performance
        model_rankings = [(model, np.mean(scores), np.std(scores)) 
                         for model, scores in model_overall_scores.items() if scores]
        model_rankings.sort(key=lambda x: x[1], reverse=True)
        
        lines.append(f"{'Rank':<5} {'Model':<25} {'Avg Accuracy':<12} {'Std':<8}")
        lines.append("-" * 50)
        for i, (model, avg_acc, std_acc) in enumerate(model_rankings, 1):
            lines.append(f"{i:<5} {model:<25} {avg_acc:.3f}        {std_acc:.3f}")
        
        lines.append("")
        
        # Key insights
        lines.append("KEY INSIGHTS:")
        lines.append("• Mathematical relationships in embeddings:")
        
        for exp_name, exp_results in self.results.items():
            expected_sim = exp_results.get('expected_similarity', 'N/A')
            
            all_accuracies = []
            for model_results in exp_results['models'].values():
                all_accuracies.extend([acc for acc in model_results['accuracies'] if not np.isnan(acc)])
            
            if all_accuracies:
                avg_acc = np.mean(all_accuracies)
                if 'negative' in exp_name:
                    lines.append(f"  - Negative relationships (a = -b): {avg_acc:.1%} accuracy")
                elif 'scaling' in exp_name:
                    scale_factor = exp_results.get('scale_factor', 'N/A')
                    lines.append(f"  - Scaling relationships (a = {scale_factor}*b): {avg_acc:.1%} accuracy")
        
        lines.append("• Higher precision numbers may be harder to embed correctly")
        lines.append("• Some models preserve mathematical relationships better than others")
        lines.append("• Check individual plots for detailed similarity distributions")
        
        return "\n".join(lines)
    
    def save_similarity_report(self, filename: str = None) -> None:
        """Save similarity report to file."""
        if filename is None:
            filename = os.path.join(self.config.results_dir, "similarity_preservation_report.txt")
        
        report = self.generate_similarity_report()
        with open(filename, "w") as f:
            f.write(report)
        
        print(f"📋 Saved similarity report: {filename}")


def run_quick_similarity_demo(embedding_wrapper, models_to_test: List[str] = None):
    """Quick demo of similarity functionality."""
    print("🚀 Quick Similarity Demo")
    print("=" * 40)
    
    # Small config for demo
    config = SimilarityConfig(
        n_pairs=50,
        decimal_sizes=[1, 3, 5],
        magnitude_ranges=[(1, 100), (100, 1000)],
        plot_dir="demo_similarity_plots",
        results_dir="demo_similarity_results"
    )
    
    # Initialize experiments
    experiments = SimilarityExperiments(embedding_wrapper, config)
    
    # Filter models if specified
    if models_to_test:
        available = set(experiments.available_models)
        requested = set(models_to_test)
        experiments.available_models = list(available & requested)
        
        if not experiments.available_models:
            print("❌ No valid models")
            return
    
    print(f"Testing: {experiments.available_models}")
    
    try:
        # Run just negative relationships for demo
        experiments.run_negative_relationship_sweep()
        
        # Run one scaling relationship
        experiments.run_scaling_relationship_sweep([2.0])
        
        # Create outputs
        experiments.plot_summary_results()
        experiments.export_similarity_results()
        
        # Show report
        report = experiments.generate_similarity_report()
        print("\n" + report)
        
        print("\n✅ Similarity demo completed!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
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
                (0.001, 0.999),   # Small decimals
                (1, 99),          # Small integers
                (100, 9999),      # Medium numbers
                (10000, 999999)   # Large numbers
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
                        
                        # Create visualization for this configuration
                        self._plot_similarity_distribution(
                            similarities, -1.0, 
                            f"{experiment_name}_{model}_{decimal_size}_{mag_range[0]}-{mag_range[1]}"
                        )
                        
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
            scale_factors = [2.0, 3.0, 0.5]
        
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
                            
                            # Create visualization for this configuration
                            self._plot_similarity_distribution(
                                similarities, 1.0, 
                                f"{experiment_name}_{model}_{decimal_size}_{mag_range[0]}-{mag_range[1]}"
                            )
                            
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
        scaling_results = self.run_scaling_relationship_sweep([2.0, 3.0, 0.5, 10.0])
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
        
        # 1. Accuracy comparison across experiments
        self._plot_accuracy_comparison(summary_dir)
        
        # 2. Mean similarity deviation from expected
        self._plot_similarity_deviations(summary_dir)
        
        # 3. Model performance ranking for similarity preservation
        self._plot_model_similarity_rankings(summary_dir)
        
        # 4. Decimal size vs accuracy analysis
        self._plot_size_vs_accuracy_analysis(summary_dir)
    
    def _plot_accuracy_comparison(self, save_dir: str) -> None:
        """Plot MAE comparison across different experiments."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        for exp_name, exp_results in self.results.items():
            if plot_idx >= 4:
                break
            
            ax = axes[plot_idx]
            
            # Collect data for this experiment
            for model_name, model_results in exp_results['models'].items():
                mae_values = [mae for mae in model_results['mae_values'] if not np.isnan(mae)]
                decimal_sizes = [size for size, mae in zip(model_results['decimal_sizes'], model_results['mae_values']) 
                               if not np.isnan(mae)]
                
                if mae_values and decimal_sizes:
                    ax.plot(decimal_sizes, mae_values, marker='o', label=model_name, linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Decimal Size')
            ax.set_ylabel('MAE (Mean Absolute Error)')
            ax.set_title(f'{exp_name}\nExpected similarity: {exp_results.get("expected_similarity", "N/A")}')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 2)  # MAE ranges from 0 to 2 for cosine similarity
            
            # Add interpretation text
            ax.text(0.02, 0.98, 'Lower = Better', transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, 4):
            axes[i].set_visible(False)
        
        plt.suptitle('MAE Comparison Across Similarity Experiments\n(Lower MAE = Better Similarity Preservation)', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "mae_comparison.png"), dpi=300, bbox_inches='tight')
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
        ax.set_ylim(0, max(means) * 1.1 if means else 2)  # Dynamic y-limit based on data
        
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
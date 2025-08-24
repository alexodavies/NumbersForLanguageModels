#!/usr/bin/env python3
"""
Improved Multi-Provider Embedding Number Reconstruction Experiment

Enhanced version with:
- Model-specific subfolders for plots
- Comprehensive CSV export of all results
- Better organization of outputs
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

# Import local modules
from datasets import (
    real_positive_decimals, 
    real_positive_and_negative_decimals, 
    real_int_and_decimal
)
from api_wrapper import EmbeddingWrapper


def create_simple_integer_dataset(n_samples: int, max_value: int = 1000) -> List[str]:
    """Create a simple dataset of integers."""
    import random
    random.seed(42)
    return [str(random.randint(1, max_value)) for _ in range(n_samples)]


def evaluate_model_simple(wrapper: EmbeddingWrapper,
                         texts: List[str],
                         true_values: List[float],
                         model: str,
                         test_size: float = 0.2) -> Dict[str, Any]:
    """
    Simplified evaluation with three core metrics.
    
    Returns:
        Dictionary with PCA and linear reconstruction results
    """
    print(f"  Evaluating {model}...")
    
    # Train/test split
    train_texts, test_texts, train_values, test_values = train_test_split(
        texts, true_values, test_size=test_size, random_state=42, shuffle=True
    )
    
    # Get embeddings
    train_embeddings = wrapper.embed(train_texts, model)
    test_embeddings = wrapper.embed(test_texts, model)
    
    X_train = np.array(train_embeddings)
    X_test = np.array(test_embeddings)
    y_train = np.array(train_values)
    y_test = np.array(test_values)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {
        'model': model,
        'train_size': len(train_texts),
        'test_size': len(test_texts),
        'embedding_dim': X_train.shape[1]
    }
    
    # 1. PCA Analysis - First Component Only
    pca = PCA(n_components=1)
    train_pca = pca.fit_transform(X_train_scaled)
    test_pca = pca.transform(X_test_scaled)
    
    # PCA explained variance ratio
    pca_explained_var_ratio = pca.explained_variance_ratio_[0]
    
    # Reconstruction R² from first PCA component
    train_pca_r2 = r2_score(y_train, train_pca.flatten())
    test_pca_r2 = r2_score(y_test, test_pca.flatten())
    
    # Try flipping if negative
    if test_pca_r2 < 0:
        test_pca_r2 = r2_score(y_test, -test_pca.flatten())
        train_pca_r2 = r2_score(y_train, -train_pca.flatten())
        test_pca = -test_pca
        train_pca = -train_pca
    
    results.update({
        'pca_explained_variance_ratio': pca_explained_var_ratio,
        'pca_train_r2': train_pca_r2,
        'pca_test_r2': test_pca_r2,
        'train_pca_component': train_pca.flatten(),
        'test_pca_component': test_pca.flatten(),
        'train_values': y_train,
        'test_values': y_test
    })
    
    # 2. Linear Model Reconstruction
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    
    train_linear_pred = ridge.predict(X_train_scaled)
    test_linear_pred = ridge.predict(X_test_scaled)
    
    train_linear_r2 = r2_score(y_train, train_linear_pred)
    test_linear_r2 = r2_score(y_test, test_linear_pred)
    
    results.update({
        'linear_train_r2': train_linear_r2,
        'linear_test_r2': test_linear_r2,
        'train_linear_pred': train_linear_pred,
        'test_linear_pred': test_linear_pred
    })
    
    print(f"    PCA Explained Var Ratio: {pca_explained_var_ratio:.3f}")
    print(f"    PCA Test R²: {test_pca_r2:.3f}, Linear Test R²: {test_linear_r2:.3f}")
    
    return results


def plot_model_results(results: Dict[str, Any], dataset_name: str, base_save_dir: str = "plots") -> None:
    """Create plots for a single model's results in model-specific subfolder."""
    model = results['model']
    
    # Create model-specific subfolder
    safe_model = model.replace('/', '_').replace('-', '_').replace(':', '_')
    model_save_dir = os.path.join(base_save_dir, safe_model)
    os.makedirs(model_save_dir, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. PCA Component vs True Values
    ax1.scatter(results['test_values'], results['test_pca_component'], 
               alpha=0.7, color='blue', label='Test', s=40)
    ax1.scatter(results['train_values'], results['train_pca_component'], 
               alpha=0.4, color='lightblue', label='Train', s=30)
    
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('First PCA Component')
    ax1.set_title(f'PCA Reconstruction\nTest R² = {results["pca_test_r2"]:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Linear Model vs True Values
    ax2.scatter(results['test_values'], results['test_linear_pred'], 
               alpha=0.7, color='green', label='Test', s=40)
    ax2.scatter(results['train_values'], results['train_linear_pred'], 
               alpha=0.4, color='lightgreen', label='Train', s=30)
    
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Linear Predictions')
    ax2.set_title(f'Linear Reconstruction\nTest R² = {results["linear_test_r2"]:.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. PCA vs Linear Comparison
    ax3.scatter(results['test_pca_component'], results['test_linear_pred'], 
               alpha=0.7, color='purple', s=40)
    ax3.set_xlabel('PCA Component 1')
    ax3.set_ylabel('Linear Predictions')
    ax3.set_title('PCA vs Linear Predictions')
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance Summary
    ax4.axis('off')
    
    # Display key metrics as text
    ax4.text(0.1, 0.9, f"Model: {model}", fontweight='bold', fontsize=11, transform=ax4.transAxes)
    ax4.text(0.1, 0.75, f"Dataset: {dataset_name}", fontweight='bold', fontsize=10, transform=ax4.transAxes)
    ax4.text(0.1, 0.6, f"Embedding Dimension: {results['embedding_dim']}", fontsize=10, transform=ax4.transAxes)
    ax4.text(0.1, 0.45, f"PCA Explained Variance: {results['pca_explained_variance_ratio']:.3f}", 
             fontsize=11, transform=ax4.transAxes)
    ax4.text(0.1, 0.3, f"PCA Test R²: {results['pca_test_r2']:.3f}", fontsize=11, transform=ax4.transAxes)
    ax4.text(0.1, 0.15, f"Linear Test R²: {results['linear_test_r2']:.3f}", fontsize=11, transform=ax4.transAxes)
    
    ax4.set_title('Experiment Summary')
    
    plt.suptitle(f'{dataset_name} - {model}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot in model-specific subfolder
    safe_dataset = dataset_name.replace('/', '_').replace(' ', '_')
    filename = f"{safe_dataset}.png"
    plt.savefig(os.path.join(model_save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()


class SimplifiedReconstructionExperiment:
    """Improved experiment class with enhanced outputs."""
    
    def __init__(self, 
                 openai_api_key: str = None,
                 google_api_key: str = None, 
                 voyage_api_key: str = None):
        """Initialize the experiment."""
        self.wrapper = EmbeddingWrapper(
            openai_api_key=openai_api_key,
            google_api_key=google_api_key,
            voyage_api_key=voyage_api_key
        )
        self.results = {}
        self.detailed_results = []  # For CSV export
        
        # Get available models
        self.all_models = []
        available_services = self.wrapper.get_available_services()
        supported_models = self.wrapper.get_supported_models()
        
        for service, is_available in available_services.items():
            if is_available:
                self.all_models.extend(supported_models[service])
                print(f"✓ {service.title()} service available")
            else:
                print(f"✗ {service.title()} service not available")
        
        if not self.all_models:
            # Fallback for testing
            for models in supported_models.values():
                self.all_models.extend(models)
            print("⚠️  No API keys provided. Will use mock data.")
        
        print(f"\nTesting {len(self.all_models)} models: {self.all_models}")
    
    def run_experiment(self, n_samples: int = 500, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Run the improved reconstruction experiment.
        
        Args:
            n_samples: Number of samples per dataset
            test_size: Fraction for test set
            
        Returns:
            Dictionary of results
        """
        print("="*80)
        print("IMPROVED EMBEDDING RECONSTRUCTION EXPERIMENT")
        print("="*80)
        
        # Define datasets to test
        datasets = [
            ("Small_Decimals", lambda n: real_positive_decimals(n, 5)),
            ("Small_Mixed_Decimals", lambda n: real_positive_and_negative_decimals(n, 5)),
            ("Medium_Decimals", lambda n: real_positive_decimals(n, 10)),
            ("Medium_Mixed_Decimals", lambda n: real_positive_and_negative_decimals(n, 10)),
            ("Large_Decimals", lambda n: real_positive_decimals(n, 15)),
            ("Large_Mixed_Decimals", lambda n: real_positive_and_negative_decimals(n, 15)),
            ("Balanced_Small", lambda n: real_int_and_decimal(n, 5, 5)),
            ("Balanced_Medium", lambda n: real_int_and_decimal(n, 15, 15)),
            ("Balanced_Large", lambda n: real_int_and_decimal(n, 15, 15)),
            ("Simple_Integers", lambda n: create_simple_integer_dataset(n, 1000)),
        ]
        
        all_results = {}
        
        for dataset_name, dataset_generator in datasets:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name}")
            print('='*60)
            
            # Generate dataset
            texts = dataset_generator(n_samples)
            true_values = [float(x) for x in texts]
            
            print(f"Value range: {min(true_values):.3f} to {max(true_values):.3f}")
            
            dataset_results = {}
            
            for model in self.all_models:
                try:
                    model_results = evaluate_model_simple(
                        self.wrapper, texts, true_values, model, test_size
                    )
                    dataset_results[model] = model_results
                    
                    # Add to detailed results for CSV
                    self._add_to_detailed_results(dataset_name, model_results)
                    
                    # Create individual plot in model subfolder
                    plot_model_results(model_results, dataset_name)
                    
                except Exception as e:
                    print(f"    Error with {model}: {str(e)}")
                    dataset_results[model] = {'error': str(e)}
            
            all_results[dataset_name] = dataset_results
        
        self.results = all_results
        return all_results
    
    def _add_to_detailed_results(self, dataset_name: str, result: Dict[str, Any]) -> None:
        """Add result to detailed results list for CSV export."""
        row = {
            'dataset': dataset_name,
            'model': result['model'],
            'train_size': result['train_size'],
            'test_size': result['test_size'],
            'embedding_dim': result['embedding_dim'],
            'pca_explained_variance_ratio': result['pca_explained_variance_ratio'],
            'pca_train_r2': result['pca_train_r2'],
            'pca_test_r2': result['pca_test_r2'],
            'linear_train_r2': result['linear_train_r2'],
            'linear_test_r2': result['linear_test_r2'],
        }
        
        # Add dataset statistics
        train_values = result['train_values']
        test_values = result['test_values']
        all_values = np.concatenate([train_values, test_values])
        
        row.update({
            'dataset_min_value': float(np.min(all_values)),
            'dataset_max_value': float(np.max(all_values)),
            'dataset_mean_value': float(np.mean(all_values)),
            'dataset_std_value': float(np.std(all_values)),
            'dataset_range': float(np.max(all_values) - np.min(all_values))
        })
        
        self.detailed_results.append(row)
    
    def export_to_csv(self, filename: str = "embedding_reconstruction_results.csv") -> None:
        """Export all results to a comprehensive CSV file."""
        if not self.detailed_results:
            print("No results to export")
            return
        
        df = pd.DataFrame(self.detailed_results)
        
        # Sort by model and dataset for better organization
        df = df.sort_values(['model', 'dataset']).reset_index(drop=True)
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"📊 Exported {len(df)} results to {filename}")
        
        # Print summary statistics
        print("\nCSV Export Summary:")
        print(f"  Total experiments: {len(df)}")
        print(f"  Models tested: {df['model'].nunique()}")
        print(f"  Datasets tested: {df['dataset'].nunique()}")
        print(f"  Average PCA Test R²: {df['pca_test_r2'].mean():.3f} ± {df['pca_test_r2'].std():.3f}")
        print(f"  Average Linear Test R²: {df['linear_test_r2'].mean():.3f} ± {df['linear_test_r2'].std():.3f}")
    
    def create_summary_plots(self, save_dir: str = "plots") -> None:
        """Create summary comparison plots."""
        if not self.results:
            print("No results to plot")
            return
        
        print("\nCreating summary plots...")
        
        # Create summary subfolder
        summary_dir = os.path.join(save_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        
        # 1. Overall Performance Heatmap
        self._plot_performance_heatmap(summary_dir)
        
        # 2. Method Comparison
        self._plot_method_comparison(summary_dir)
        
        # 3. Model Rankings
        self._plot_model_rankings(summary_dir)
        
        # 4. Dataset Difficulty Analysis
        self._plot_dataset_difficulty(summary_dir)
    
    def _plot_performance_heatmap(self, save_dir: str) -> None:
        """Create performance heatmap across datasets and models."""
        datasets = list(self.results.keys())
        models = list(set(model for dataset_results in self.results.values() 
                         for model in dataset_results.keys() 
                         if 'error' not in dataset_results[model]))
        
        # Create matrices for PCA and Linear test R²
        pca_matrix = np.full((len(datasets), len(models)), np.nan)
        linear_matrix = np.full((len(datasets), len(models)), np.nan)
        
        for i, dataset in enumerate(datasets):
            for j, model in enumerate(models):
                if (model in self.results[dataset] and 
                    'error' not in self.results[dataset][model]):
                    result = self.results[dataset][model]
                    pca_matrix[i, j] = result.get('pca_test_r2', np.nan)
                    linear_matrix[i, j] = result.get('linear_test_r2', np.nan)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # PCA heatmap
        im1 = ax1.imshow(pca_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        ax1.set_title('PCA Test R² Scores', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels([m.replace('-', '\n') for m in models], rotation=45, ha='right', fontsize=10)
        ax1.set_yticks(range(len(datasets)))
        ax1.set_yticklabels(datasets, fontsize=10)
        
        # Add text annotations
        for i in range(len(datasets)):
            for j in range(len(models)):
                if not np.isnan(pca_matrix[i, j]):
                    ax1.text(j, i, f'{pca_matrix[i, j]:.2f}',
                            ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im1, ax=ax1, label='Test R² Score')
        
        # Linear heatmap
        im2 = ax2.imshow(linear_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        ax2.set_title('Linear Test R² Scores', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels([m.replace('-', '\n') for m in models], rotation=45, ha='right', fontsize=10)
        ax2.set_yticks(range(len(datasets)))
        ax2.set_yticklabels(datasets, fontsize=10)
        
        # Add text annotations
        for i in range(len(datasets)):
            for j in range(len(models)):
                if not np.isnan(linear_matrix[i, j]):
                    ax2.text(j, i, f'{linear_matrix[i, j]:.2f}',
                            ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im2, ax=ax2, label='Test R² Score')
        
        plt.suptitle('Performance Heatmaps Across All Models and Datasets', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_dir, "performance_heatmaps.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_method_comparison(self, save_dir: str) -> None:
        """Compare PCA vs Linear methods."""
        pca_scores = []
        linear_scores = []
        labels = []
        
        for dataset, dataset_results in self.results.items():
            for model, result in dataset_results.items():
                if 'error' not in result:
                    pca_scores.append(result['pca_test_r2'])
                    linear_scores.append(result['linear_test_r2'])
                    labels.append(f"{dataset}\n{model}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        ax1.scatter(pca_scores, linear_scores, alpha=0.7, s=60, color='darkblue')
        max_score = max(max(pca_scores), max(linear_scores))
        min_score = min(min(pca_scores), min(linear_scores))
        ax1.plot([min_score, max_score], [min_score, max_score], 'r--', alpha=0.8, label='Equal Performance')
        ax1.set_xlabel('PCA Test R²', fontsize=12)
        ax1.set_ylabel('Linear Test R²', fontsize=12)
        ax1.set_title('PCA vs Linear Performance Comparison', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plots
        bp = ax2.boxplot([pca_scores, linear_scores], labels=['PCA', 'Linear'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightgreen')
        ax2.set_ylabel('Test R² Score', fontsize=12)
        ax2.set_title('Method Performance Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "method_comparison.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_rankings(self, save_dir: str) -> None:
        """Plot model rankings by average performance."""
        # Calculate average scores per model
        model_scores = {}
        for dataset_results in self.results.values():
            for model, result in dataset_results.items():
                if 'error' not in result:
                    if model not in model_scores:
                        model_scores[model] = {'pca': [], 'linear': []}
                    model_scores[model]['pca'].append(result['pca_test_r2'])
                    model_scores[model]['linear'].append(result['linear_test_r2'])
        
        # Calculate averages
        model_avg = {}
        for model, scores in model_scores.items():
            model_avg[model] = {
                'pca_avg': np.mean(scores['pca']),
                'linear_avg': np.mean(scores['linear']),
                'pca_std': np.std(scores['pca']),
                'linear_std': np.std(scores['linear'])
            }
        
        # Sort by linear performance
        sorted_models = sorted(model_avg.items(), key=lambda x: x[1]['linear_avg'], reverse=True)
        
        models = [item[0] for item in sorted_models]
        pca_avgs = [item[1]['pca_avg'] for item in sorted_models]
        linear_avgs = [item[1]['linear_avg'] for item in sorted_models]
        pca_stds = [item[1]['pca_std'] for item in sorted_models]
        linear_stds = [item[1]['linear_std'] for item in sorted_models]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, pca_avgs, width, yerr=pca_stds, 
                      label='PCA', alpha=0.8, capsize=5, color='skyblue')
        bars2 = ax.bar(x + width/2, linear_avgs, width, yerr=linear_stds, 
                      label='Linear', alpha=0.8, capsize=5, color='lightgreen')
        
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Average Test R² Score', fontsize=12)
        ax.set_title('Model Performance Rankings (Average Across All Datasets)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('-', '\n') for m in models], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "model_rankings.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_dataset_difficulty(self, save_dir: str) -> None:
        """Plot dataset difficulty analysis."""
        dataset_scores = {}
        
        for dataset, dataset_results in self.results.items():
            dataset_scores[dataset] = []
            for model, result in dataset_results.items():
                if 'error' not in result:
                    dataset_scores[dataset].append(result['linear_test_r2'])
        
        # Calculate average difficulty (lower score = harder dataset)
        dataset_difficulty = {dataset: np.mean(scores) for dataset, scores in dataset_scores.items()}
        sorted_datasets = sorted(dataset_difficulty.items(), key=lambda x: x[1])
        
        datasets = [item[0] for item in sorted_datasets]
        avg_scores = [item[1] for item in sorted_datasets]
        std_scores = [np.std(dataset_scores[dataset]) for dataset in datasets]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.bar(range(len(datasets)), avg_scores, yerr=std_scores, 
                     capsize=5, alpha=0.8, color='coral')
        
        ax.set_xlabel('Datasets (Ordered by Difficulty)', fontsize=12)
        ax.set_ylabel('Average Linear Test R² Score', fontsize=12)
        ax.set_title('Dataset Difficulty Analysis\n(Lower scores indicate more challenging datasets)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (avg, std) in enumerate(zip(avg_scores, std_scores)):
            ax.text(i, avg + std + 0.01, f'{avg:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "dataset_difficulty.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report."""
        lines = []
        lines.append("IMPROVED EMBEDDING RECONSTRUCTION EXPERIMENT RESULTS")
        lines.append("="*70)
        
        if not self.results:
            lines.append("No results to report.")
            return "\n".join(lines)
        
        # Overall statistics
        all_exp_var_ratios = []
        all_pca_scores = []
        all_linear_scores = []
        
        for dataset_results in self.results.values():
            for result in dataset_results.values():
                if 'error' not in result:
                    all_exp_var_ratios.append(result['pca_explained_variance_ratio'])
                    all_pca_scores.append(result['pca_test_r2'])
                    all_linear_scores.append(result['linear_test_r2'])
        
        lines.append(f"Total successful experiments: {len(all_pca_scores)}")
        lines.append(f"Models tested: {len(set(r['model'] for dr in self.results.values() for r in dr.values() if 'error' not in r))}")
        lines.append(f"Datasets tested: {len(self.results)}")
        lines.append("")
        lines.append("OVERALL PERFORMANCE STATISTICS:")
        lines.append("-" * 40)
        lines.append(f"PCA Explained Variance Ratio - Mean: {np.mean(all_exp_var_ratios):.3f}, Std: {np.std(all_exp_var_ratios):.3f}")
        lines.append(f"PCA Test R² - Mean: {np.mean(all_pca_scores):.3f}, Std: {np.std(all_pca_scores):.3f}")
        lines.append(f"Linear Test R² - Mean: {np.mean(all_linear_scores):.3f}, Std: {np.std(all_linear_scores):.3f}")
        lines.append("")
        
        # Top performers
        best_results = []
        for dataset, dataset_results in self.results.items():
            for model, result in dataset_results.items():
                if 'error' not in result:
                    best_results.append((dataset, model, result['pca_explained_variance_ratio'], 
                                       result['pca_test_r2'], result['linear_test_r2']))
        
        best_results.sort(key=lambda x: x[4], reverse=True)  # Sort by linear R²
        
        lines.append("TOP 10 PERFORMERS (Linear Test R²):")
        lines.append("-" * 80)
        lines.append(f"{'Dataset':<20} {'Model':<25} {'Exp Var':<8} {'PCA R²':<8} {'Linear R²':<10}")
        lines.append("-" * 80)
        for i, (dataset, model, exp_var, pca_r2, linear_r2) in enumerate(best_results[:10]):
            lines.append(f"{dataset:<20} {model:<25} {exp_var:.3f}    {pca_r2:.3f}    {linear_r2:.3f}")
        
        lines.append("")
        lines.append("WORST 5 PERFORMERS (Linear Test R²):")
        lines.append("-" * 80)
        lines.append(f"{'Dataset':<20} {'Model':<25} {'Exp Var':<8} {'PCA R²':<8} {'Linear R²':<10}")
        lines.append("-" * 80)
        for i, (dataset, model, exp_var, pca_r2, linear_r2) in enumerate(best_results[-5:]):
            lines.append(f"{dataset:<20} {model:<25} {exp_var:.3f}    {pca_r2:.3f}    {linear_r2:.3f}")
        
        lines.append("")
        
        # Model rankings
        model_scores = {}
        for dataset_results in self.results.values():
            for model, result in dataset_results.items():
                if 'error' not in result:
                    if model not in model_scores:
                        model_scores[model] = []
                    model_scores[model].append(result['linear_test_r2'])
        
        model_averages = [(model, np.mean(scores), np.std(scores)) 
                         for model, scores in model_scores.items()]
        model_averages.sort(key=lambda x: x[1], reverse=True)
        
        lines.append("MODEL RANKINGS (Average Linear Test R²):")
        lines.append("-" * 50)
        lines.append(f"{'Rank':<5} {'Model':<25} {'Mean R²':<10} {'Std R²':<8}")
        lines.append("-" * 50)
        for i, (model, mean_score, std_score) in enumerate(model_averages, 1):
            lines.append(f"{i:<5} {model:<25} {mean_score:.3f}     {std_score:.3f}")
        
        lines.append("")
        
        # Dataset difficulty analysis
        dataset_scores = {}
        for dataset, dataset_results in self.results.items():
            dataset_scores[dataset] = []
            for model, result in dataset_results.items():
                if 'error' not in result:
                    dataset_scores[dataset].append(result['linear_test_r2'])
        
        dataset_difficulty = [(dataset, np.mean(scores), np.std(scores)) 
                             for dataset, scores in dataset_scores.items()]
        dataset_difficulty.sort(key=lambda x: x[1])  # Sort by difficulty (lowest score = hardest)
        
        lines.append("DATASET DIFFICULTY RANKING (Average Linear Test R²):")
        lines.append("-" * 55)
        lines.append(f"{'Rank':<5} {'Dataset':<20} {'Mean R²':<10} {'Std R²':<8} {'Difficulty':<12}")
        lines.append("-" * 55)
        for i, (dataset, mean_score, std_score) in enumerate(dataset_difficulty, 1):
            difficulty = "Very Hard" if mean_score < 0.3 else "Hard" if mean_score < 0.6 else "Medium" if mean_score < 0.8 else "Easy"
            lines.append(f"{i:<5} {dataset:<20} {mean_score:.3f}     {std_score:.3f}    {difficulty:<12}")
        
        return "\n".join(lines)


def main():
    """Main execution function."""
    print("Improved Multi-Provider Embedding Reconstruction Experiment")
    print("=" * 70)
    
    # Get API keys
    openai_key = os.getenv('OPENAI_API_KEY')
    google_key = os.getenv('GOOGLE_API_KEY') 
    voyage_key = os.getenv('VOYAGE_API_KEY')
    
    if not any([openai_key, google_key, voyage_key]):
        print("⚠️  No API keys found. Set OPENAI_API_KEY, GOOGLE_API_KEY, and/or VOYAGE_API_KEY")
        print("The experiment will run with mock data.\n")
    
    try:
        # Initialize experiment
        experiment = SimplifiedReconstructionExperiment(
            openai_api_key=openai_key,
            google_api_key=google_key,
            voyage_api_key=voyage_key
        )
        
        # Run main experiment
        print("\n🔬 Running improved reconstruction experiment...")
        results = experiment.run_experiment(n_samples=500, test_size=0.2)
        
        # Export to CSV
        print("\n📊 Exporting results to CSV...")
        experiment.export_to_csv("embedding_reconstruction_results.csv")
        
        # Create summary plots
        print("\n📈 Creating summary plots...")
        experiment.create_summary_plots()
        
        # Generate report
        print("\n📋 Generating comprehensive report...")
        report = experiment.generate_summary_report()
        print("\n" + report)
        
        # Save report
        with open("comprehensive_results_report.txt", "w") as f:
            f.write(report)
        
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("Generated Files and Directories:")
        print("  📄 comprehensive_results_report.txt - Detailed text report")
        print("  📊 embedding_reconstruction_results.csv - Complete results data")
        print("  📁 plots/")
        print("    📁 [model_name]/ - Individual model result plots")
        print("      📈 [dataset_name].png - Dataset-specific plots for each model")
        print("    📁 summary/ - Aggregate analysis plots")
        print("      📈 performance_heatmaps.png - Cross-model performance comparison")
        print("      📈 method_comparison.png - PCA vs Linear method analysis")
        print("      📈 model_rankings.png - Average model performance rankings")
        print("      📈 dataset_difficulty.png - Dataset difficulty analysis")
        print("\n📋 Summary Statistics:")
        
        # Quick summary
        if experiment.detailed_results:
            df = pd.DataFrame(experiment.detailed_results)
            print(f"  ✅ Total experiments completed: {len(df)}")
            print(f"  🤖 Unique models tested: {df['model'].nunique()}")
            print(f"  📊 Unique datasets tested: {df['dataset'].nunique()}")
            print(f"  🎯 Best Linear R² achieved: {df['linear_test_r2'].max():.3f}")
            print(f"  📈 Average Linear R² across all experiments: {df['linear_test_r2'].mean():.3f}")
            
            # Find best performing model overall
            best_model = df.groupby('model')['linear_test_r2'].mean().idxmax()
            best_score = df.groupby('model')['linear_test_r2'].mean().max()
            print(f"  🏆 Best performing model: {best_model} (avg R² = {best_score:.3f})")
        
        print("\n💡 Tips for Analysis:")
        print("  • Check the CSV file for detailed quantitative analysis")
        print("  • Review individual model plots to understand per-dataset performance")
        print("  • Use summary plots to identify patterns across models and datasets")
        print("  • Dataset difficulty ranking helps understand which number types are hardest to embed")
        
    except Exception as e:
        print(f"❌ Experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())